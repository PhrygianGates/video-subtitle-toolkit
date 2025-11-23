#!/usr/bin/env python3
"""
Translate SRT subtitle files using OpenAI-compatible APIs.

This script translates SRT subtitle content while preserving all formatting,
timestamps, and structure.

Environment variables required:
- NV_API_KEY (for NVIDIA API, default provider)
- AZURE_OPENAI_API_KEY (for Azure OpenAI, when using --api-provider azure)

Usage:
  # Using NVIDIA API (default)
  python translate_srt.py input.srt --target zh
  
  # Using Azure OpenAI
  python translate_srt.py input.srt --target zh --api-provider azure
  
  # Specify output file
  python translate_srt.py input.srt --target en --output output.srt
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


# ------------------------------ OpenAI Client ------------------------------

def get_openai_client(api_provider: str = "nv"):
    """
    Return an OpenAI client instance based on the provider.
    
    Args:
        api_provider: Either "azure" or "nv" (NVIDIA)
    
    Returns:
        OpenAI client instance
    """
    try:
        from openai import OpenAI, AzureOpenAI  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency: pip install openai>=1.0.0"
        ) from exc
    
    if api_provider == "nv":
        # NVIDIA API configuration
        api_key = os.environ.get("NV_API_KEY")
        if not api_key:
            raise RuntimeError(
                "NV_API_KEY must be set as environment variable when using --api-provider nv."
            )
        
        return OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
    
    elif api_provider == "azure":
        # Azure OpenAI configuration
        endpoint = "https://llm-proxy.perflab.nvidia.com"
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        api_version = "2025-02-01-preview"
        
        if not api_key:
            raise RuntimeError(
                "AZURE_OPENAI_API_KEY must be set as environment variable when using --api-provider azure."
            )
        
        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version
        )
    
    else:
        raise ValueError(f"Unknown API provider: {api_provider}. Must be 'azure' or 'nv'.")


def chat_complete(client, model: str, system_prompt: str, user_content: str) -> str:
    """Call OpenAI Chat Completions API and return the text."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
        top_p=1.0,
    )
    text = resp.choices[0].message.content or ""
    return text.strip()


# ------------------------------ SRT Processing ------------------------------

@dataclass
class SubtitleBlock:
    """Represents a single subtitle block in an SRT file."""
    index: str
    timestamp: str
    content: str
    
    def __str__(self) -> str:
        """Convert back to SRT format."""
        return f"{self.index}\n{self.timestamp}\n{self.content}\n"


def parse_srt(srt_text: str) -> List[SubtitleBlock]:
    """Parse SRT content into SubtitleBlock objects."""
    blocks = []
    
    # Split by double newlines (or more) to separate subtitle blocks
    raw_blocks = re.split(r'\n\s*\n', srt_text.strip())
    
    for raw_block in raw_blocks:
        if not raw_block.strip():
            continue
            
        lines = raw_block.strip().split('\n')
        
        if len(lines) < 3:
            # Malformed block, skip
            continue
        
        # First line: index
        index = lines[0].strip()
        
        # Second line: timestamp (format: HH:MM:SS,mmm --> HH:MM:SS,mmm)
        timestamp = lines[1].strip()
        
        # Check if this is a valid timestamp line
        if '-->' not in timestamp:
            # Malformed timestamp, skip
            continue
        
        # Remaining lines: content
        content = '\n'.join(lines[2:])
        
        blocks.append(SubtitleBlock(
            index=index,
            timestamp=timestamp,
            content=content
        ))
    
    return blocks


def build_translation_prompt(target_lang: str) -> str:
    """Build system prompt for subtitle translation."""
    lang_map = {
        "zh": "Simplified Chinese (zh-Hans)",
        "en": "English",
        "ja": "Japanese",
        "ko": "Korean",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
    }
    
    target_name = lang_map.get(target_lang, target_lang)
    
    return (
        f"You are a precise subtitle translator. Translate each subtitle line to {target_name}.\n\n"
        "CRITICAL RULES:\n"
        "1. Lines marked [CONTEXT] are for reference only - DO NOT translate or include them in output.\n"
        "2. Lines marked [TRANSLATE #N] MUST be translated - translate each one separately and independently.\n"
        "3. Output MUST be a valid JSON array of objects with EXACTLY the same number of elements as [TRANSLATE #N] lines.\n"
        "4. Each object has two fields: \"id\" (the number N from [TRANSLATE #N]) and \"text\" (the translation).\n"
        "5. Count carefully - if you see [TRANSLATE #1] through [TRANSLATE #50], output 50 objects with id 1-50.\n"
        "6. Keep each subtitle concise and independent - suitable for on-screen display.\n"
        "7. Maintain the original structure - do not reorganize, combine, or rewrite.\n"
        "8. DO NOT include any text outside the JSON array - no explanations, no markdown code blocks.\n\n"
        "Example Format:\n"
        "Input:\n"
        "[CONTEXT] This is context for reference only\n"
        "[TRANSLATE #1] Hello world\n"
        "[TRANSLATE #2] How are you\n"
        "[TRANSLATE #3] Goodbye\n"
        "[CONTEXT] More context after\n\n"
        "Output (pure JSON array with numbered objects, no markdown):\n"
        '[{\"id\": 1, \"text\": \"你好世界\"}, {\"id\": 2, \"text\": \"你好吗\"}, {\"id\": 3, \"text\": \"再见\"}]\n\n'
        "NOTE: Count the [TRANSLATE #N] lines - output exactly that many objects with sequential IDs. [CONTEXT] lines are NOT included."
    )


def translate_subtitle_batch(
    client,
    model: str,
    blocks: List[SubtitleBlock],
    target_lang: str,
    context_before: List[SubtitleBlock],
    context_after: List[SubtitleBlock]
) -> List[str]:
    """
    Translate a batch of subtitle blocks with context.
    
    Args:
        client: Azure OpenAI client
        model: Model name
        blocks: List of subtitle blocks to translate
        target_lang: Target language code
        context_before: Context blocks before the batch (for reference only)
        context_after: Context blocks after the batch (for reference only)
    
    Returns:
        List of translated content strings
    """
    system_prompt = build_translation_prompt(target_lang)
    
    # Build the user content with context markers and numbering
    lines = []
    translate_count = 0
    
    # Add context before
    for block in context_before:
        lines.append(f"[CONTEXT] {block.content}")
    
    # Add blocks to translate with explicit numbering
    for idx, block in enumerate(blocks, 1):
        translate_count = idx
        lines.append(f"[TRANSLATE #{idx}] {block.content}")
    
    # Add context after
    for block in context_after:
        lines.append(f"[CONTEXT] {block.content}")
    
    user_content = "\n\n".join(lines)
    
    # Add a reminder at the end
    user_content += f'\n\n[REMINDER: Count carefully! Output a JSON array with exactly {translate_count} objects. Format: [{{"id": 1, "text": "translation1"}}, {{"id": 2, "text": "translation2"}}, ...]]'
    
    # Get translation
    translated = chat_complete(client, model, system_prompt, user_content)
    
    # Try to parse as JSON first
    translated_blocks = []
    try:
        # Remove markdown code blocks if present
        json_text = translated.strip()
        if json_text.startswith("```json"):
            json_text = json_text[7:]
        if json_text.startswith("```"):
            json_text = json_text[3:]
        if json_text.endswith("```"):
            json_text = json_text[:-3]
        json_text = json_text.strip()
        
        # Parse JSON
        json_data = json.loads(json_text)
        
        # Ensure it's a list
        if not isinstance(json_data, list):
            raise ValueError("JSON output is not an array")
        
        # Check if it's the new format with id/text objects
        if json_data and isinstance(json_data[0], dict) and 'id' in json_data[0] and 'text' in json_data[0]:
            # New format: array of {"id": N, "text": "translation"} objects
            # Sort by id to ensure correct order
            json_data.sort(key=lambda x: x.get('id', 0))
            
            # Verify IDs are sequential
            expected_ids = list(range(1, len(json_data) + 1))
            actual_ids = [item.get('id') for item in json_data]
            
            if actual_ids != expected_ids:
                print(f"  ⚠ Warning: IDs not sequential. Expected {expected_ids[:3]}...{expected_ids[-2:]}, got {actual_ids[:3]}...{actual_ids[-2:]}")
            
            # Extract text fields
            translated_blocks = [str(item.get('text', '')) for item in json_data]
            print(f"  ✓ Successfully parsed JSON with {len(translated_blocks)} numbered translations", end=' ')
        else:
            # Old format: simple string array
            translated_blocks = [str(item) for item in json_data]
            print(f"  ✓ Successfully parsed JSON with {len(translated_blocks)} translations", end=' ')
        
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  ✗ JSON parsing failed: {e}")
        print(f"  → Falling back to text splitting...")
        
        # Fallback: Try to split the translation into individual blocks
        # First try double newlines, then single newlines
        translated_blocks = [t.strip() for t in translated.split("\n\n") if t.strip()]
        
        # If double newline split doesn't give us the right count, try single newline
        if len(translated_blocks) != len(blocks):
            translated_blocks = [t.strip() for t in translated.split("\n") if t.strip()]
    
    # Handle mismatch between expected and actual number of translations
    if len(translated_blocks) != len(blocks):
        print(f"\n{'='*80}")
        print(f"WARNING: Translation count mismatch!")
        print(f"Expected: {len(blocks)} translations (IDs should be 1 to {len(blocks)})")
        print(f"Got: {len(translated_blocks)} translations")
        print(f"{'='*80}")
        
        # Show first 2 and last 2 blocks for debugging
        print("\n--- ORIGINAL TEXT (first 2) ---")
        for i in range(min(2, len(blocks))):
            print(f"[{i+1}] {blocks[i].content[:100]}...")
        
        print("\n--- ORIGINAL TEXT (last 2) ---")
        for i in range(max(0, len(blocks)-2), len(blocks)):
            print(f"[{i+1}] {blocks[i].content[:100]}...")
        
        print("\n--- TRANSLATED TEXT (first 2) ---")
        for i in range(min(2, len(translated_blocks))):
            print(f"[{i+1}] {translated_blocks[i][:100]}...")
        
        print("\n--- TRANSLATED TEXT (last 2) ---")
        for i in range(max(0, len(translated_blocks)-2), len(translated_blocks)):
            print(f"[{i+1}] {translated_blocks[i][:100]}...")
        
        print(f"{'='*80}\n")
        
        if len(translated_blocks) < len(blocks):
            # Pad with original content for missing translations
            print(f"Padding with original content for {len(blocks) - len(translated_blocks)} missing blocks")
            translated_blocks.extend([blocks[i].content for i in range(len(translated_blocks), len(blocks))])
        elif len(translated_blocks) > len(blocks):
            # Truncate extra translations
            print(f"Truncating {len(translated_blocks) - len(blocks)} extra translations")
            translated_blocks = translated_blocks[:len(blocks)]
    
    return translated_blocks


def translate_srt_file(
    client,
    model: str,
    input_path: Path,
    output_path: Path,
    target_lang: str,
    batch_size: int = 10,
    context_size: int = 1
) -> None:
    """
    Translate an entire SRT file with batch processing and context.
    
    Args:
        client: Azure OpenAI client
        model: Model name
        input_path: Input SRT file path
        output_path: Output SRT file path
        target_lang: Target language code (zh, en, ja, etc.)
        batch_size: Number of subtitle blocks to translate at once
        context_size: Number of context blocks to provide before/after each batch
    """
    # Read input file
    srt_text = input_path.read_text(encoding="utf-8")
    
    # Parse SRT blocks
    blocks = parse_srt(srt_text)
    
    if not blocks:
        print(f"No valid subtitle blocks found in {input_path}", file=sys.stderr)
        return
    
    print(f"Found {len(blocks)} subtitle blocks")
    print(f"Translating in batches of {batch_size} with {context_size} context blocks...")
    
    # Translate in batches
    translated_blocks = []
    total_batches = (len(blocks) + batch_size - 1) // batch_size
    
    for batch_idx in range(0, len(blocks), batch_size):
        batch_num = batch_idx // batch_size + 1
        print(f"Translating batch {batch_num}/{total_batches}...", end=' ')
        
        # Get current batch
        batch_end = min(batch_idx + batch_size, len(blocks))
        current_batch = blocks[batch_idx:batch_end]
        
        # Get context before
        context_start = max(0, batch_idx - context_size)
        context_before = blocks[context_start:batch_idx] if batch_idx > 0 else []
        
        # Get context after
        context_after_end = min(len(blocks), batch_end + context_size)
        context_after = blocks[batch_end:context_after_end] if batch_end < len(blocks) else []
        
        try:
            # Translate batch with context
            translated_contents = translate_subtitle_batch(
                client=client,
                model=model,
                blocks=current_batch,
                target_lang=target_lang,
                context_before=context_before,
                context_after=context_after
            )
            
            # Create translated blocks
            for i, content in enumerate(translated_contents):
                translated_blocks.append(SubtitleBlock(
                    index=current_batch[i].index,
                    timestamp=current_batch[i].timestamp,
                    content=content
                ))
            print()  # New line after successful batch
        except Exception as e:
            print(f"\n✗ Error translating batch {batch_num}: {e}", file=sys.stderr)
            print(f"Keeping original content for this batch.", file=sys.stderr)
            # Keep original content on error
            translated_blocks.extend(current_batch)
    
    print(f"\nTranslation complete!")
    
    # Write output file
    output_content = '\n'.join(str(block) for block in translated_blocks)
    output_path.write_text(output_content, encoding="utf-8")
    
    print(f"Wrote translated subtitles to: {output_path}")


# ------------------------------ Main ------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate SRT subtitle files using OpenAI-compatible APIs"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input SRT file path"
    )
    parser.add_argument(
        "--target",
        "-t",
        required=True,
        help="Target language code (zh, en, ja, ko, es, fr, de, etc.)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output SRT file path (default: input_file.{target_lang}.srt)"
    )
    parser.add_argument(
        "--api-provider",
        "-p",
        choices=["nv", "azure"],
        default="nv",
        help="API provider to use: 'nv' (NVIDIA) or 'azure' (Azure OpenAI) (default: nv)"
    )
    parser.add_argument(
        "--model",
        default="deepseek-ai/deepseek-v3.1",
        help="Model name to use (default: deepseek-ai/deepseek-v3.1)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of subtitle blocks to translate at once (default: 10)"
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=1,
        help="Number of context blocks before/after each batch (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    # Determine output file
    if args.output:
        output_path = Path(args.output)
    else:
        # Generate output filename: input.target_lang.srt
        output_path = input_path.parent / f"{input_path.stem}.{args.target}{input_path.suffix}"
    
    # Initialize OpenAI client
    try:
        client = get_openai_client(api_provider=args.api_provider)
        print(f"Using API provider: {args.api_provider}")
        print(f"Using model: {args.model}")
    except (RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Translate
    translate_srt_file(
        client=client,
        model=args.model,
        input_path=input_path,
        output_path=output_path,
        target_lang=args.target,
        batch_size=args.batch_size,
        context_size=args.context_size
    )


if __name__ == "__main__":
    main()

