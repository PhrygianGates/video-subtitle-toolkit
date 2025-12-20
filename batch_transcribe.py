#!/usr/bin/env python3
"""
Batch transcribe and translate video files with pipeline parallelism.

This script processes all video files in a directory with intelligent parallel processing:
1. Extract audio using ffmpeg (piped to whisper, no temp files)
2. Transcribe audio using whisper.cpp with VAD (Voice Activity Detection)
   - Automatic segmentation (default: 16-minute segments) to avoid hallucinations
   - VAD enabled to detect silence and improve accuracy
3. Translate subtitles using translate_srt.py (high concurrency for API calls)

Pipeline parallelism allows:
- Video A: transcribing (CPU-intensive, serialized)
- Video B: translating (I/O-intensive, parallel)

Usage:
  python batch_transcribe.py <video_directory> [options]
  
Example:
  python batch_transcribe.py "TTC【双语字幕版】：认识地球 - 地质学导论（上）" --target zh
"""

import argparse
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Semaphore
from typing import List, Optional, Tuple


# Common video file extensions
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v'}

# Default paths (can be overridden with arguments)
DEFAULT_WHISPER_BIN = os.path.expanduser("~/Development/whisper.cpp/build/bin/whisper-cli")
DEFAULT_WHISPER_MODEL = os.path.expanduser("~/Development/whisper.cpp/models/ggml-large-v2.bin")
DEFAULT_VAD_MODEL = os.path.expanduser("~/Development/whisper.cpp/models/ggml-silero-v6.2.0.bin")

# Default processing settings
DEFAULT_SEGMENT_DURATION = 16  # minutes
DEFAULT_MAX_WORKERS = 2

# Thread-safe printing
print_lock = threading.Lock()

# Global semaphore to control transcription concurrency (set during runtime)
transcribe_semaphore = None


def safe_print(*args, **kwargs):
    """Thread-safe print function."""
    with print_lock:
        print(*args, **kwargs)


def find_video_files(directory: Path, pattern: Optional[str] = None) -> List[Path]:
    """
    Find video files in the directory, optionally filtered by regex pattern.
    
    Args:
        directory: Directory to search
        pattern: Optional regex pattern to match against filenames
    
    Returns:
        List of matching video file paths
    """
    video_files = []
    regex = re.compile(pattern) if pattern else None
    
    for file_path in sorted(directory.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in VIDEO_EXTENSIONS:
            # Apply regex filter if provided
            if regex and not regex.search(file_path.name):
                continue
            video_files.append(file_path)
    
    return video_files


def get_video_duration(video_path: Path) -> Optional[float]:
    """
    Get video duration in seconds using ffprobe.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Duration in seconds, or None if unable to determine
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        return duration
    except (subprocess.CalledProcessError, ValueError) as e:
        return None


def parse_srt_timestamp(timestamp: str) -> float:
    """Convert SRT timestamp to seconds."""
    # Format: HH:MM:SS,mmm
    time_part, ms_part = timestamp.split(',')
    h, m, s = map(int, time_part.split(':'))
    ms = int(ms_part)
    return h * 3600 + m * 60 + s + ms / 1000


def format_srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def merge_srt_files(srt_paths: List[Path], output_path: Path, segment_offsets: List[float]) -> bool:
    """
    Merge multiple SRT files with time offset adjustments.
    
    Args:
        srt_paths: List of SRT file paths to merge (in order)
        output_path: Output merged SRT file path
        segment_offsets: Time offsets in seconds for each segment
    
    Returns:
        True if successful, False otherwise
    """
    try:
        merged_entries = []
        entry_number = 1
        
        for srt_path, offset in zip(srt_paths, segment_offsets):
            if not srt_path.exists():
                continue
                
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Split into subtitle entries
            entries = content.split('\n\n')
            
            for entry in entries:
                if not entry.strip():
                    continue
                
                lines = entry.strip().split('\n')
                if len(lines) < 3:
                    continue
                
                # Parse timestamp line (second line)
                timestamp_line = lines[1]
                if '-->' not in timestamp_line:
                    continue
                
                start_str, end_str = timestamp_line.split(' --> ')
                
                # Add offset to timestamps
                start_seconds = parse_srt_timestamp(start_str.strip()) + offset
                end_seconds = parse_srt_timestamp(end_str.strip()) + offset
                
                # Create new entry with adjusted timestamps
                new_entry = f"{entry_number}\n"
                new_entry += f"{format_srt_timestamp(start_seconds)} --> {format_srt_timestamp(end_seconds)}\n"
                new_entry += '\n'.join(lines[2:])
                
                merged_entries.append(new_entry)
                entry_number += 1
        
        # Write merged file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(merged_entries))
            if merged_entries:
                f.write('\n')
        
        return True
    except Exception as e:
        print(f"Error merging SRT files: {e}", file=sys.stderr)
        return False


def extract_and_transcribe_piped(
    video_path: Path,
    whisper_bin: str,
    whisper_model: str,
    vad_model: Optional[str] = None,
    segment_duration: Optional[int] = None,
    language: str = "auto",
    thread_safe: bool = False
) -> bool:
    """
    Extract audio and transcribe using a pipe (no temporary file).
    
    Args:
        video_path: Path to video file
        whisper_bin: Path to whisper binary
        whisper_model: Path to whisper model
        vad_model: Optional path to VAD model for voice activity detection
        segment_duration: If set, split video into segments of this many minutes
        thread_safe: Whether to use thread-safe printing
    
    Returns:
        True if successful, False otherwise
    """
    print_fn = safe_print if thread_safe else print
    output_dir = video_path.parent
    base_name = video_path.stem
    srt_path = output_dir / f"{base_name}.srt"
    
    # If no segmentation or video is short, process normally
    if segment_duration is None:
        return _extract_and_transcribe_whole(
            video_path, whisper_bin, whisper_model, vad_model, language, thread_safe
        )
    
    # Get video duration
    duration = get_video_duration(video_path)
    if duration is None:
        print_fn(f"  ⚠ Could not determine video duration, processing as whole file")
        return _extract_and_transcribe_whole(
            video_path, whisper_bin, whisper_model, vad_model, language, thread_safe
        )
    
    segment_seconds = segment_duration * 60
    
    # If video is shorter than segment duration, process normally
    if duration <= segment_seconds:
        print_fn(f"  ℹ Video duration ({duration/60:.1f}min) ≤ segment size ({segment_duration}min), processing as whole")
        return _extract_and_transcribe_whole(
            video_path, whisper_bin, whisper_model, vad_model, language, thread_safe
        )
    
    # Process in segments
    print_fn(f"  ℹ Video duration: {duration/60:.1f}min, splitting into {segment_duration}min segments")
    
    num_segments = int((duration + segment_seconds - 1) // segment_seconds)
    segment_srt_paths = []
    segment_offsets = []
    
    # Use semaphore to limit concurrent transcriptions if available
    global transcribe_semaphore
    semaphore_ctx = transcribe_semaphore if (transcribe_semaphore and thread_safe) else None
    
    if semaphore_ctx:
        semaphore_ctx.acquire()
    
    try:
        for i in range(num_segments):
            start_time = i * segment_seconds
            segment_offsets.append(start_time)
            
            # Format time as HH:MM:SS
            start_str = f"{int(start_time//3600):02d}:{int((start_time%3600)//60):02d}:{int(start_time%60):02d}"
            
            print_fn(f"  📹🎤 Segment {i+1}/{num_segments}: transcribing from {start_str}")
            
            # FFmpeg command: extract audio segment to stdout
            ffmpeg_cmd = [
                "ffmpeg",
                "-ss", start_str,
                "-i", str(video_path),
                "-t", str(segment_seconds),
                "-ar", "16000",
                "-ac", "1",
                "-c:a", "pcm_s16le",
                "-f", "wav",
                "-"
            ]
            
            # Output for this segment
            segment_base = f"{base_name}_seg{i+1:03d}"
            segment_srt = output_dir / f"{segment_base}.srt"
            segment_srt_paths.append(segment_srt)
            
            # Whisper command: read from stdin
            whisper_cmd = [
                whisper_bin,
                "-m", whisper_model,
                "-l", language,
                "-",
                "-osrt",
                "-of", str(output_dir / segment_base)
            ]
            
            # Add VAD support if model is provided
            if vad_model:
                whisper_cmd.extend(["--vad", "-vm", vad_model])
            
            # Transcribe this segment
            if not _do_extract_and_transcribe(
                ffmpeg_cmd, whisper_cmd, output_dir, segment_base, print_fn
            ):
                print_fn(f"  ✗ Failed to transcribe segment {i+1}")
                return False
        
        # Merge all segment SRT files
        print_fn(f"  🔗 Merging {num_segments} segments...")
        if not merge_srt_files(segment_srt_paths, srt_path, segment_offsets):
            print_fn(f"  ✗ Failed to merge SRT files")
            return False
        
        # Clean up segment files
        print_fn(f"  🧹 Cleaning up segment files...")
        for seg_srt in segment_srt_paths:
            if seg_srt.exists():
                seg_srt.unlink()
        
        print_fn(f"  ✓ Merged transcription complete: {srt_path.name}")
        return True
        
    finally:
        if semaphore_ctx:
            semaphore_ctx.release()


def _extract_and_transcribe_whole(
    video_path: Path,
    whisper_bin: str,
    whisper_model: str,
    vad_model: Optional[str] = None,
    language: str = "auto",
    thread_safe: bool = False
) -> bool:
    """Process entire video as one unit."""
    print_fn = safe_print if thread_safe else print
    output_dir = video_path.parent
    base_name = video_path.stem
    
    # FFmpeg command: extract audio to stdout
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        "-f", "wav",
        "-"
    ]
    
    # Whisper command: read from stdin
    whisper_cmd = [
        whisper_bin,
        "-m", whisper_model,
        "-l", language,
        "-",
        "-osrt",
        "-of", str(output_dir / base_name)
    ]
    
    # Add VAD support if model is provided
    if vad_model:
        whisper_cmd.extend(["--vad", "-vm", vad_model])
    
    # Use semaphore to limit concurrent transcriptions if available
    global transcribe_semaphore
    if transcribe_semaphore and thread_safe:
        with transcribe_semaphore:
            print_fn(f"  📹🎤 Extracting and transcribing: {video_path.name}")
            return _do_extract_and_transcribe(
                ffmpeg_cmd, whisper_cmd, output_dir, base_name, print_fn
            )
    else:
        print_fn(f"  📹🎤 Extracting and transcribing: {video_path.name}")
        return _do_extract_and_transcribe(
            ffmpeg_cmd, whisper_cmd, output_dir, base_name, print_fn
        )


def _do_extract_and_transcribe(ffmpeg_cmd, whisper_cmd, output_dir, base_name, print_fn):
    """Internal function to perform the actual extraction and transcription."""
    try:
        # Start ffmpeg process
        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Start whisper process, reading from ffmpeg's stdout
        whisper_proc = subprocess.Popen(
            whisper_cmd,
            stdin=ffmpeg_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Allow ffmpeg_proc to receive a SIGPIPE if whisper_proc exits
        if ffmpeg_proc.stdout:
            ffmpeg_proc.stdout.close()
        
        # Wait for both processes to complete
        whisper_stdout, whisper_stderr = whisper_proc.communicate()
        ffmpeg_proc.wait()
        
        # Check for errors
        if ffmpeg_proc.returncode != 0:
            ffmpeg_stderr = ffmpeg_proc.stderr.read() if ffmpeg_proc.stderr else b""
            print_fn(f"  ✗ FFmpeg error (code {ffmpeg_proc.returncode})", file=sys.stderr)
            if ffmpeg_stderr:
                error_msg = ffmpeg_stderr.decode('utf-8', errors='ignore').strip()
                # Only print last few lines of error
                error_lines = error_msg.split('\n')[-5:]
                print_fn(f"  FFmpeg error: {' '.join(error_lines)}", file=sys.stderr)
            return False
        
        if whisper_proc.returncode != 0:
            print_fn(f"  ✗ Whisper error (code {whisper_proc.returncode})", file=sys.stderr)
            if whisper_stderr:
                error_msg = whisper_stderr.decode('utf-8', errors='ignore').strip()
                # Only print last few lines of error
                error_lines = error_msg.split('\n')[-5:]
                print_fn(f"  Whisper error: {' '.join(error_lines)}", file=sys.stderr)
            return False
        
        # Check if SRT file was created
        expected_srt = output_dir / f"{base_name}.srt"
        if expected_srt.exists():
            print_fn(f"  ✓ Extraction and transcription complete: {expected_srt.name}")
            return True
        else:
            print_fn(f"  ✗ Expected SRT file not found: {expected_srt.name}", file=sys.stderr)
            return False
            
    except Exception as e:
        print_fn(f"  ✗ Error in piped extraction/transcription: {e}", file=sys.stderr)
        return False


def translate_srt(
    srt_path: Path,
    target_lang: str,
    translate_script: Path,
    batch_size: int = 10,
    thread_safe: bool = False
) -> bool:
    """Translate SRT file using translate_srt.py."""
    print_fn = safe_print if thread_safe else print
    print_fn(f"  🌍 Translating subtitles to {target_lang}: {srt_path.name}")
    
    cmd = [
        sys.executable,  # Use the same Python interpreter
        str(translate_script),
        str(srt_path),
        "--target", target_lang,
        "--batch-size", str(batch_size)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True
        )
        
        # Check if translated file was created
        expected_output = srt_path.parent / f"{srt_path.stem}.{target_lang}.srt"
        if expected_output.exists():
            print_fn(f"  ✓ Translation complete: {expected_output.name}")
            return True
        else:
            print_fn(f"  ✗ Expected translated file not found: {expected_output.name}", file=sys.stderr)
            return False
    except subprocess.CalledProcessError as e:
        print_fn(f"  ✗ Error translating subtitles: {e}", file=sys.stderr)
        return False


def process_video(
    video_path: Path,
    target_lang: str,
    whisper_bin: str,
    whisper_model: str,
    translate_script: Path,
    vad_model: Optional[str] = None,
    segment_duration: Optional[int] = None,
    language: str = "auto",
    skip_existing: bool = True,
    translate_batch_size: int = 10,
    thread_safe: bool = False
) -> Tuple[str, str]:
    """
    Process a single video file: extract+transcribe (piped), then translate.
    
    Args:
        video_path: Path to video file
        target_lang: Target language for translation
        whisper_bin: Path to whisper binary
        whisper_model: Path to whisper model
        translate_script: Path to translation script
        vad_model: Optional path to VAD model for voice activity detection
        segment_duration: If set, split video into segments of this many minutes
        skip_existing: Skip if translated file already exists
        translate_batch_size: Number of subtitle blocks per translation batch
        thread_safe: Use thread-safe printing
    
    Returns:
        Tuple of (status, video_name) where status is 'success', 'skipped', or 'failed'
    """
    print_fn = safe_print if thread_safe else print
    
    print_fn(f"\n{'='*80}")
    print_fn(f"Processing: {video_path.name}")
    print_fn(f"{'='*80}")
    
    srt_path = video_path.parent / f"{video_path.stem}.srt"
    translated_srt = video_path.parent / f"{video_path.stem}.{target_lang}.srt"
    
    # Check if translated file already exists
    if skip_existing and translated_srt.exists():
        print_fn(f"  ⏭ Skipping (translation already exists): {video_path.name}")
        return ('skipped', video_path.name)
    
    # Check if SRT already exists
    if srt_path.exists():
        print_fn(f"  ℹ SRT file already exists: {srt_path.name}")
        print_fn(f"  Skipping extraction and transcription...")
    else:
        # Step 1 & 2: Extract audio and transcribe (piped, no temp file)
        if not extract_and_transcribe_piped(
            video_path, whisper_bin, whisper_model, vad_model, segment_duration, language, thread_safe
        ):
            return ('failed', video_path.name)
    
    # Step 3: Translate subtitles
    if not translate_srt(srt_path, target_lang, translate_script, translate_batch_size, thread_safe):
        return ('failed', video_path.name)
    
    print_fn(f"  ✅ Successfully processed: {video_path.name}")
    return ('success', video_path.name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch transcribe and translate video files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: 2 parallel workers, 16-minute segments, VAD enabled automatically
  # This allows pipeline processing: one video transcribing while others translating
  python batch_transcribe.py "TTC【双语字幕版】：认识地球 - 地质学导论（上）" --target zh
  
  # Process with 10-minute segments for better accuracy on problematic videos
  python batch_transcribe.py "Videos" --target zh -s 10
  
  # Disable segmentation (process videos as whole)
  python batch_transcribe.py "Videos" --target zh -s 0
  
  # Process only specific files matching a pattern (e.g., files starting with "01")
  python batch_transcribe.py "Videos" --target zh --pattern "^01_.*"
  
  # Process files containing "火山" in the name
  python batch_transcribe.py "Videos" --target zh --pattern "火山"
  
  # Process only specific video file (exact match) with segmentation
  python batch_transcribe.py "Videos" --target zh --pattern "^15_.*\\.mp4$" -s 15
  
  # High concurrency: 6 workers total, 2 can transcribe simultaneously
  python batch_transcribe.py "My Videos" --target en --max-workers 6 --max-transcribe 2
  
  # Disable VAD even if VAD model exists
  python batch_transcribe.py "Videos" --target zh --no-vad
  
  # Specify input language (default: auto)
  python batch_transcribe.py "Videos" --target zh --language en
  python batch_transcribe.py "Videos" --target zh -l fr
  
  # Adjust translation batch size (more blocks per API call = faster but may reduce quality)
  python batch_transcribe.py "Videos" --target zh --translate-batch-size 20
  python batch_transcribe.py "Videos" --target zh -tb 5  # smaller batches for better quality
  
  # Sequential processing (single worker)
  python batch_transcribe.py "Videos" --target zh --max-workers 1
  
  # Force reprocess videos even if translations exist
  python batch_transcribe.py "My Videos" --target en --no-skip-existing
  
  # Use custom whisper paths
  python batch_transcribe.py "Videos" --target zh \\
    --whisper-bin /path/to/whisper-cli \\
    --whisper-model /path/to/model.bin
"""
    )
    
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing video files"
    )
    
    parser.add_argument(
        "--target",
        "-t",
        required=True,
        help="Target language code for translation (zh, en, ja, etc.)"
    )
    
    parser.add_argument(
        "--whisper-bin",
        default=DEFAULT_WHISPER_BIN,
        help=f"Path to whisper-cli binary (default: {DEFAULT_WHISPER_BIN})"
    )
    
    parser.add_argument(
        "--whisper-model",
        default=DEFAULT_WHISPER_MODEL,
        help=f"Path to whisper model (default: {DEFAULT_WHISPER_MODEL})"
    )
    
    parser.add_argument(
        "--translate-script",
        default=None,
        help="Path to translate_srt.py (default: auto-detect in same directory)"
    )
    
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        default=True,
        help="Process videos even if translated subtitles already exist (default: skip existing)"
    )
    
    parser.add_argument(
        "--max-workers",
        "-j",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Maximum number of videos to process in parallel (default: {DEFAULT_MAX_WORKERS})"
    )
    
    parser.add_argument(
        "--max-transcribe",
        type=int,
        default=1,
        help="Maximum number of concurrent transcriptions (default: 1, recommended for CPU efficiency)"
    )
    
    parser.add_argument(
        "--pattern",
        "-p",
        type=str,
        default=None,
        help="Regex pattern to filter video files (e.g., '01_.*\\.mp4' or '火山')"
    )
    
    parser.add_argument(
        "--segment-duration",
        "-s",
        type=int,
        default=DEFAULT_SEGMENT_DURATION,
        help=f"Split videos into segments of N minutes for transcription (default: {DEFAULT_SEGMENT_DURATION}). "
             "Set to 0 to disable segmentation and process videos as whole."
    )
    
    parser.add_argument(
        "--no-vad",
        action="store_true",
        default=False,
        help="Disable Voice Activity Detection even if VAD model is available"
    )
    
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default="auto",
        help="Input language for whisper transcription (default: auto). "
             "Use language codes like: en, zh, ja, fr, de, etc."
    )
    
    parser.add_argument(
        "--translate-batch-size",
        "-tb",
        type=int,
        default=10,
        help="Number of subtitle blocks to translate per API call (default: 10). "
             "Larger batches are faster but may reduce translation quality."
    )
    
    args = parser.parse_args()
    
    # Validate directory
    video_dir = Path(args.directory)
    if not video_dir.exists():
        print(f"Error: Directory not found: {video_dir}", file=sys.stderr)
        sys.exit(1)
    
    if not video_dir.is_dir():
        print(f"Error: Not a directory: {video_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Find translate_srt.py
    if args.translate_script:
        translate_script = Path(args.translate_script)
    else:
        # Auto-detect in the same directory as this script
        translate_script = Path(__file__).parent / "translate_srt.py"
    
    if not translate_script.exists():
        print(f"Error: translate_srt.py not found: {translate_script}", file=sys.stderr)
        print("Please specify the path using --translate-script", file=sys.stderr)
        sys.exit(1)
    
    # Validate whisper binary
    whisper_bin = Path(args.whisper_bin).expanduser()
    if not whisper_bin.exists():
        print(f"Error: whisper-cli not found: {whisper_bin}", file=sys.stderr)
        print("Please specify the path using --whisper-bin", file=sys.stderr)
        sys.exit(1)
    
    # Validate whisper model
    whisper_model = Path(args.whisper_model).expanduser()
    if not whisper_model.exists():
        print(f"Error: whisper model not found: {whisper_model}", file=sys.stderr)
        print("Please specify the path using --whisper-model", file=sys.stderr)
        sys.exit(1)
    
    # Check VAD model (use if exists and not disabled by user)
    vad_model_path = Path(DEFAULT_VAD_MODEL).expanduser()
    if args.no_vad:
        vad_model = None
        print(f"VAD: Disabled by user (--no-vad)")
    elif vad_model_path.exists():
        vad_model = str(vad_model_path)
    else:
        vad_model = None
        print(f"Warning: VAD model not found: {vad_model_path}", file=sys.stderr)
        print("Voice Activity Detection will be disabled. Transcription quality may be reduced.", file=sys.stderr)
    
    # Handle segment duration (0 means disable)
    segment_duration = args.segment_duration if args.segment_duration > 0 else None
    
    # Validate regex pattern if provided
    if args.pattern:
        try:
            re.compile(args.pattern)
        except re.error as e:
            print(f"Error: Invalid regex pattern: {args.pattern}", file=sys.stderr)
            print(f"Regex error: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Find all video files (with optional pattern filtering)
    video_files = find_video_files(video_dir, args.pattern)
    
    if not video_files:
        if args.pattern:
            print(f"No video files matching pattern '{args.pattern}' found in: {video_dir}", file=sys.stderr)
        else:
            print(f"No video files found in: {video_dir}", file=sys.stderr)
        print(f"Supported extensions: {', '.join(sorted(VIDEO_EXTENSIONS))}", file=sys.stderr)
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"Batch Transcription and Translation")
    print(f"{'='*80}")
    print(f"Directory: {video_dir}")
    if args.pattern:
        print(f"Filter pattern: {args.pattern}")
    print(f"Video files found: {len(video_files)}")
    if args.pattern and video_files:
        print(f"Matched files:")
        for vf in video_files:
            print(f"  - {vf.name}")
    print(f"Input language: {args.language}")
    print(f"Target language: {args.target}")
    print(f"Whisper binary: {whisper_bin}")
    print(f"Whisper model: {whisper_model}")
    if vad_model:
        print(f"VAD model: {vad_model} (voice activity detection enabled)")
    elif args.no_vad:
        print(f"VAD: Disabled by user (--no-vad)")
    else:
        print(f"VAD: Disabled (model not found)")
    print(f"Translation script: {translate_script}")
    print(f"Using piped processing (no temp files)")
    if segment_duration:
        print(f"Segmentation: {segment_duration} minutes per segment")
    else:
        print(f"Segmentation: Disabled (process videos as whole)")
    print(f"Skip existing: {'Yes' if args.skip_existing else 'No'}")
    print(f"Max parallel workers: {args.max_workers}")
    print(f"Max concurrent transcriptions: {args.max_transcribe}")
    print(f"Translation batch size: {args.translate_batch_size} blocks per API call")
    print(f"{'='*80}")
    
    # Initialize transcription semaphore for limiting concurrent transcriptions
    global transcribe_semaphore
    transcribe_semaphore = Semaphore(args.max_transcribe)
    
    # Process each video
    success_count = 0
    skip_count = 0
    failed_count = 0
    
    # Use ThreadPoolExecutor for parallel processing
    use_parallel = args.max_workers > 1
    
    if use_parallel:
        print(f"\n🚀 Processing videos in parallel with {args.max_workers} workers...\n")
        
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit all video processing tasks
            future_to_video = {
                executor.submit(
                    process_video,
                    video_path,
                    args.target,
                    str(whisper_bin),
                    str(whisper_model),
                    translate_script,
                    vad_model,
                    segment_duration,
                    args.language,
                    args.skip_existing,
                    args.translate_batch_size,
                    True  # thread_safe=True for parallel processing
                ): (i, video_path)
                for i, video_path in enumerate(video_files, 1)
            }
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_video):
                i, video_path = future_to_video[future]
                try:
                    status, video_name = future.result()
                    if status == 'success':
                        success_count += 1
                    elif status == 'skipped':
                        skip_count += 1
                    elif status == 'failed':
                        failed_count += 1
                except Exception as e:
                    safe_print(f"✗ Exception processing {video_path.name}: {e}", file=sys.stderr)
                    failed_count += 1
    else:
        print(f"\n📼 Processing videos sequentially...\n")
        
        for i, video_path in enumerate(video_files, 1):
            print(f"[{i}/{len(video_files)}]")
            
            status, video_name = process_video(
                video_path,
                args.target,
                str(whisper_bin),
                str(whisper_model),
                translate_script,
                vad_model,
                segment_duration,
                args.language,
                args.skip_existing,
                args.translate_batch_size,
                False  # thread_safe=False for sequential processing
            )
            
            if status == 'success':
                success_count += 1
            elif status == 'skipped':
                skip_count += 1
            elif status == 'failed':
                failed_count += 1
    
    # Summary
    print(f"\n{'='*80}")
    print(f"Batch Processing Complete!")
    print(f"{'='*80}")
    print(f"Total videos: {len(video_files)}")
    print(f"Successfully processed: {success_count}")
    print(f"Skipped: {skip_count}")
    print(f"Failed: {failed_count}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

