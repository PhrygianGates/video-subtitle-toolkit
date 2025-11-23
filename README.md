# Video Subtitle Toolkit

A powerful toolkit for automated video transcription and subtitle translation, designed for batch processing educational content and lecture videos.

## 🌟 Features

- **Batch Video Transcription**: Extract audio and transcribe using whisper.cpp
  - Automatic audio extraction via ffmpeg (piped, no temp files)
  - Voice Activity Detection (VAD) for improved accuracy
  - Smart segmentation to prevent hallucinations on long videos
  - Pipeline parallelism: transcribe while translating
  
- **Intelligent Subtitle Translation**: Translate SRT files using OpenAI-compatible APIs
  - Context-aware batch translation
  - Support for multiple AI providers (NVIDIA, Azure OpenAI)
  - Preserves timing and formatting
  - High concurrency for fast processing

## 📋 Requirements

### System Dependencies
- Python 3.7+
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) (compiled with binaries)
- ffmpeg and ffprobe

### Python Dependencies
```bash
pip install openai
```

### Whisper Models
Download whisper models from [whisper.cpp models](https://github.com/ggerganov/whisper.cpp/tree/master/models):
- Speech recognition model (e.g., `ggml-large-v2.bin`)
- VAD model (optional but recommended): `ggml-silero-v5.1.2.bin`

## 🚀 Quick Start

### 1. Transcribe Videos

```bash
# Basic usage: transcribe all videos in a directory and translate to Chinese
python batch_transcribe.py "path/to/videos" --target zh

# With custom settings
python batch_transcribe.py "path/to/videos" --target zh \
  --segment-duration 10 \
  --max-workers 2

# Process specific files with pattern matching
python batch_transcribe.py "path/to/videos" --target zh --pattern "^01_.*"
```

### 2. Translate Subtitles Only

```bash
# Using NVIDIA API (default)
export NV_API_KEY="your-api-key"
python translate_srt.py input.srt --target zh

# Using Azure OpenAI
export AZURE_OPENAI_API_KEY="your-api-key"
python translate_srt.py input.srt --target zh --api-provider azure
```

## 📖 Usage Guide

### `batch_transcribe.py`

Batch process video files with transcription and translation:

```bash
python batch_transcribe.py <video_directory> [options]
```

**Key Options:**
- `--target, -t`: Target language code (zh, en, ja, etc.) - **required**
- `--segment-duration, -s`: Split videos into N-minute segments (default: 16)
- `--max-workers, -j`: Number of videos to process in parallel (default: 2)
- `--max-transcribe`: Maximum concurrent transcriptions (default: 1)
- `--pattern, -p`: Regex pattern to filter video files
- `--whisper-bin`: Path to whisper-cli binary
- `--whisper-model`: Path to whisper model file
- `--no-skip-existing`: Force reprocessing of existing files

**Examples:**

```bash
# Process with 10-minute segments for better accuracy
python batch_transcribe.py "Videos" --target zh -s 10

# High concurrency: 6 workers, 2 transcribing simultaneously
python batch_transcribe.py "Videos" --target en --max-workers 6 --max-transcribe 2

# Process only files containing "火山" in the name
python batch_transcribe.py "Videos" --target zh --pattern "火山"
```

### `translate_srt.py`

Translate SRT subtitle files:

```bash
python translate_srt.py <input.srt> --target <language> [options]
```

**Key Options:**
- `--target, -t`: Target language code - **required**
- `--output, -o`: Output file path (default: `input.{target}.srt`)
- `--api-provider, -p`: API provider: `nv` (NVIDIA) or `azure` (default: nv)
- `--model`: Model name (default: deepseek-ai/deepseek-v3.1)
- `--batch-size`: Subtitles per batch (default: 10)
- `--context-size`: Context blocks before/after batch (default: 1)

**Examples:**

```bash
# Translate to English with custom output
python translate_srt.py video.srt --target en --output video_english.srt

# Use Azure OpenAI with larger batches
python translate_srt.py video.srt --target zh \
  --api-provider azure \
  --batch-size 20
```

## ⚙️ Configuration

### Whisper Paths (in `batch_transcribe.py`)

Default paths can be overridden with command-line arguments:
```python
DEFAULT_WHISPER_BIN = "~/Development/whisper.cpp/build/bin/whisper-cli"
DEFAULT_WHISPER_MODEL = "~/Development/whisper.cpp/models/ggml-large-v2.bin"
DEFAULT_VAD_MODEL = "~/Development/whisper.cpp/models/ggml-silero-v5.1.2.bin"
```

### API Endpoints (in `translate_srt.py`)

⚠️ You may need to modify the API endpoints in the code.


### Environment Variables

For `translate_srt.py`:
- `NV_API_KEY`: NVIDIA API key (when using `--api-provider nv`)
- `AZURE_OPENAI_API_KEY`: Azure OpenAI key (when using `--api-provider azure`)

## 🎯 Supported Languages

Translation supports common languages via OpenAI-compatible APIs:
- `zh` - Simplified Chinese
- `en` - English
- `ja` - Japanese
- `ko` - Korean
- `es` - Spanish
- `fr` - French
- `de` - German
- And more...

## 🏗️ Architecture

### Pipeline Parallelism

The toolkit uses intelligent parallelism:
1. **Transcription** (CPU-intensive): Serialized by default to avoid CPU contention
2. **Translation** (I/O-intensive): High concurrency for API calls

This allows Video A to transcribe while Video B translates, maximizing throughput.

### Segmentation Strategy

Long videos are automatically split into segments:
- Prevents hallucinations common in whisper models
- Default: 16-minute segments
- Each segment is transcribed independently, then merged with correct timestamps

## 📂 Output Structure

For a video `lecture.mp4`:
```
lecture.mp4                 # Original video
lecture.srt                 # Transcribed subtitles (original language)
lecture.zh.srt             # Translated subtitles (Chinese)
```

## 📝 License

This project is provided as-is for personal and educational use.


