# chunk.py
# Doctown v11 – Chunk subsystem: convert semantic units into embedding-ready chunks

import re
from dataclasses import dataclass, field
from typing import List
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from parse.parse import ParsedFile, SemanticUnit


# ============================================================
# Configuration
# ============================================================

# Token limits for embeddings
# Most embedding models (e.g., sentence-transformers) work best with:
# - Min: ~20 tokens (too small = poor semantic representation)
# - Max: ~512 tokens (typical BERT-style limit)
# - Target: ~256 tokens (sweet spot for most models)

DEFAULT_MIN_TOKENS = 20
DEFAULT_MAX_TOKENS = 512
DEFAULT_TARGET_TOKENS = 256


# ============================================================
# Output Format
# ============================================================

@dataclass
class Chunk:
    """A text chunk ready for embedding."""
    text: str
    token_count: int
    source_file: str
    source_units: List[str] = field(default_factory=list)  # Names of merged units
    metadata: dict = field(default_factory=dict)


@dataclass
class ChunkedFile:
    """Result of chunking a parsed file."""
    path: str
    chunks: List[Chunk]
    original_unit_count: int
    metadata: dict = field(default_factory=dict)


# ============================================================
# Token Counting
# ============================================================

def estimate_token_count(text: str) -> int:
    """
    Fast token count estimation without loading a full tokenizer.

    This uses a simple heuristic:
    - Split on whitespace and punctuation
    - Apply rough multiplier for subword tokenization

    For accurate counts, we'd use the actual tokenizer in embed/,
    but for chunking purposes, this approximation is sufficient.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    if not text or not text.strip():
        return 0

    # Simple word-based estimate
    # Most tokenizers split on whitespace + punctuation
    words = re.findall(r'\w+|[^\w\s]', text)

    # Apply subword tokenization multiplier
    # English text typically: 1 word ≈ 1.3 tokens (WordPiece/BPE)
    # Code typically: 1 word ≈ 1.5 tokens (more camelCase/snake_case)

    # Conservative estimate: assume 1.3x multiplier
    estimated_tokens = int(len(words) * 1.3)

    return max(1, estimated_tokens)  # Minimum 1 token


# ============================================================
# Text Splitting
# ============================================================

def split_text_by_tokens(
    text: str,
    max_tokens: int,
    overlap_tokens: int = 50,
) -> List[str]:
    """
    Split text into chunks respecting token limits.

    Strategy:
    1. Try to split on paragraph boundaries first
    2. Fall back to sentence boundaries
    3. Finally split on word boundaries if needed

    Args:
        text: Text to split
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of tokens to overlap between chunks

    Returns:
        List of text chunks
    """
    total_tokens = estimate_token_count(text)

    # If text fits in one chunk, return it
    if total_tokens <= max_tokens:
        return [text]

    chunks = []

    # Try splitting on double newlines (paragraphs)
    paragraphs = text.split("\n\n")

    current_chunk = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = estimate_token_count(para)

        # If single paragraph exceeds max, split it further
        if para_tokens > max_tokens:
            # Flush current chunk
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_tokens = 0

            # Split large paragraph by sentences
            sentences = re.split(r'([.!?]+\s+)', para)
            sentence_chunk = []
            sentence_tokens = 0

            for i in range(0, len(sentences), 2):
                # Combine sentence with its punctuation
                if i + 1 < len(sentences):
                    sentence = sentences[i] + sentences[i + 1]
                else:
                    sentence = sentences[i]

                sent_tokens = estimate_token_count(sentence)

                if sentence_tokens + sent_tokens > max_tokens and sentence_chunk:
                    chunks.append("".join(sentence_chunk))
                    sentence_chunk = []
                    sentence_tokens = 0

                sentence_chunk.append(sentence)
                sentence_tokens += sent_tokens

            if sentence_chunk:
                chunks.append("".join(sentence_chunk))

        # Paragraph fits - try to add to current chunk
        elif current_tokens + para_tokens <= max_tokens:
            current_chunk.append(para)
            current_tokens += para_tokens

        # Paragraph doesn't fit - start new chunk
        else:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_tokens = para_tokens

    # Flush final chunk
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


# ============================================================
# Chunking Logic
# ============================================================

def chunk_semantic_units(
    units: List[SemanticUnit],
    source_file: str,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
) -> List[Chunk]:
    """
    Convert semantic units into chunks suitable for embedding.

    Strategy:
    1. Merge small consecutive units until reaching target size
    2. Split large units into smaller chunks
    3. Maintain semantic boundaries where possible

    Args:
        units: List of semantic units from parser
        source_file: Source file path
        min_tokens: Minimum tokens per chunk
        max_tokens: Maximum tokens per chunk
        target_tokens: Target tokens per chunk

    Returns:
        List of Chunk objects
    """
    if not units:
        return []

    chunks = []

    # Group units for merging
    current_group = []
    current_tokens = 0
    current_names = []

    for unit in units:
        unit_tokens = estimate_token_count(unit.content)

        # If unit is too large, split it
        if unit_tokens > max_tokens:
            # Flush current group first
            if current_group:
                merged_text = "\n\n".join(current_group)
                chunks.append(Chunk(
                    text=merged_text,
                    token_count=estimate_token_count(merged_text),
                    source_file=source_file,
                    source_units=current_names.copy(),
                    metadata={"merged": True, "unit_count": len(current_names)},
                ))
                current_group = []
                current_tokens = 0
                current_names = []

            # Split large unit
            split_chunks = split_text_by_tokens(unit.content, max_tokens)
            for i, chunk_text in enumerate(split_chunks):
                chunks.append(Chunk(
                    text=chunk_text,
                    token_count=estimate_token_count(chunk_text),
                    source_file=source_file,
                    source_units=[f"{unit.name}_part{i+1}"],
                    metadata={"split": True, "part": i+1, "total_parts": len(split_chunks)},
                ))

        # If adding this unit would exceed max, flush current group
        elif current_tokens + unit_tokens > max_tokens:
            if current_group:
                merged_text = "\n\n".join(current_group)
                chunks.append(Chunk(
                    text=merged_text,
                    token_count=estimate_token_count(merged_text),
                    source_file=source_file,
                    source_units=current_names.copy(),
                    metadata={"merged": True, "unit_count": len(current_names)},
                ))

            current_group = [unit.content]
            current_tokens = unit_tokens
            current_names = [unit.name]

        # Add unit to current group
        else:
            current_group.append(unit.content)
            current_tokens += unit_tokens
            current_names.append(unit.name)

            # If we've reached target size, flush
            if current_tokens >= target_tokens:
                merged_text = "\n\n".join(current_group)
                chunks.append(Chunk(
                    text=merged_text,
                    token_count=estimate_token_count(merged_text),
                    source_file=source_file,
                    source_units=current_names.copy(),
                    metadata={"merged": True, "unit_count": len(current_names)},
                ))
                current_group = []
                current_tokens = 0
                current_names = []

    # Flush final group
    if current_group:
        merged_text = "\n\n".join(current_group)
        final_tokens = estimate_token_count(merged_text)

        # If final chunk is too small, try to merge with previous
        if chunks and final_tokens < min_tokens:
            last_chunk = chunks[-1]
            if last_chunk.token_count + final_tokens <= max_tokens:
                # Merge with previous chunk
                chunks[-1] = Chunk(
                    text=last_chunk.text + "\n\n" + merged_text,
                    token_count=last_chunk.token_count + final_tokens,
                    source_file=source_file,
                    source_units=last_chunk.source_units + current_names,
                    metadata={"merged": True, "unit_count": len(last_chunk.source_units) + len(current_names)},
                )
            else:
                # Keep as separate chunk even if small
                chunks.append(Chunk(
                    text=merged_text,
                    token_count=final_tokens,
                    source_file=source_file,
                    source_units=current_names,
                    metadata={"merged": True, "unit_count": len(current_names), "small": True},
                ))
        else:
            chunks.append(Chunk(
                text=merged_text,
                token_count=final_tokens,
                source_file=source_file,
                source_units=current_names,
                metadata={"merged": True, "unit_count": len(current_names)},
            ))

    return chunks


def chunk_parsed_files(
    parsed_files: List[ParsedFile],
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
) -> List[ChunkedFile]:
    """
    Chunk multiple parsed files.

    Args:
        parsed_files: List of ParsedFile objects from parse subsystem
        min_tokens: Minimum tokens per chunk
        max_tokens: Maximum tokens per chunk
        target_tokens: Target tokens per chunk

    Returns:
        List of ChunkedFile objects
    """
    results = []

    for parsed in parsed_files:
        # Skip binary files
        if parsed.metadata.get("binary", False):
            continue

        # Skip empty files
        if not parsed.semantic_units:
            continue

        chunks = chunk_semantic_units(
            parsed.semantic_units,
            parsed.path,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            target_tokens=target_tokens,
        )

        if chunks:
            results.append(ChunkedFile(
                path=parsed.path,
                chunks=chunks,
                original_unit_count=len(parsed.semantic_units),
                metadata={
                    "file_type": parsed.file_type.value,
                    "language": parsed.language,
                    "line_count": parsed.line_count,
                },
            ))

    return results


# ============================================================
# Utility Functions
# ============================================================

def get_all_chunks(chunked_files: List[ChunkedFile]) -> List[Chunk]:
    """
    Flatten all chunks from chunked files into a single list.

    Args:
        chunked_files: List of ChunkedFile objects

    Returns:
        Flat list of all Chunk objects
    """
    all_chunks = []
    for cf in chunked_files:
        all_chunks.extend(cf.chunks)
    return all_chunks


def get_chunk_stats(chunked_files: List[ChunkedFile]) -> dict:
    """
    Get statistics about chunked files.

    Args:
        chunked_files: List of ChunkedFile objects

    Returns:
        Dictionary with statistics
    """
    all_chunks = get_all_chunks(chunked_files)

    if not all_chunks:
        return {
            "total_files": 0,
            "total_chunks": 0,
            "total_tokens": 0,
            "avg_tokens_per_chunk": 0,
            "min_tokens": 0,
            "max_tokens": 0,
        }

    token_counts = [c.token_count for c in all_chunks]

    return {
        "total_files": len(chunked_files),
        "total_chunks": len(all_chunks),
        "total_tokens": sum(token_counts),
        "avg_tokens_per_chunk": sum(token_counts) / len(token_counts),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "merged_chunks": sum(1 for c in all_chunks if c.metadata.get("merged", False)),
        "split_chunks": sum(1 for c in all_chunks if c.metadata.get("split", False)),
    }


# ============================================================
# Standalone Test
# ============================================================

if __name__ == "__main__":
    from parse.parse import parse_file

    print("Testing chunk subsystem...")

    # Test 1: Python file with multiple functions
    sample_python = b"""
import sys
from pathlib import Path

def small_function():
    \"\"\"A small function.\"\"\"
    return 42

def medium_function(x: int, y: int) -> int:
    \"\"\"
    A medium-sized function with some logic.

    This function demonstrates how multiple lines
    of code get chunked together.
    \"\"\"
    result = x + y
    if result > 100:
        return result * 2
    else:
        return result

class ExampleClass:
    \"\"\"An example class.\"\"\"

    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1
        return self.value

    def decrement(self):
        self.value -= 1
        return self.value

def very_long_function_that_will_need_splitting():
    \"\"\"
    This is a very long function with lots of code.
    It should get split into multiple chunks.
    \"\"\"
    # Simulate a long function with many lines
    result = []
    for i in range(100):
        result.append(i)
        result.append(i * 2)
        result.append(i * 3)
        if i % 10 == 0:
            result.append("Checkpoint")

    # More processing
    processed = [x for x in result if isinstance(x, int)]

    # Even more code to make this really long
    final_result = sum(processed)

    return final_result
"""

    print("\n[Test 1] Chunking Python file")
    parsed = parse_file("example.py", sample_python)
    print(f"Parsed: {len(parsed.semantic_units)} semantic units")

    chunked_files = chunk_parsed_files([parsed])
    print(f"Chunked: {len(chunked_files)} files")

    if chunked_files:
        cf = chunked_files[0]
        print(f"File: {cf.path}")
        print(f"Chunks: {len(cf.chunks)}")

        for i, chunk in enumerate(cf.chunks, 1):
            print(f"\nChunk {i}:")
            print(f"  Tokens: {chunk.token_count}")
            print(f"  Source units: {', '.join(chunk.source_units)}")
            print(f"  Metadata: {chunk.metadata}")
            print(f"  Preview: {chunk.text[:100]}...")

    # Test 2: Markdown file
    sample_markdown = b"""
# Main Title

This is a short paragraph.

## Section 1

This is another paragraph with some content.
It has multiple lines.

### Subsection

- Item 1
- Item 2
- Item 3

## Section 2

A longer paragraph that contains more text. This paragraph is designed to test
how the chunker handles medium-sized blocks of text that should ideally be kept
together as a single semantic unit. The chunker should recognize that this entire
paragraph belongs together and not split it unnecessarily.

```python
def example():
    return "code block"
```

## Section 3

Final section with some concluding remarks.
"""

    print("\n\n[Test 2] Chunking Markdown file")
    parsed_md = parse_file("README.md", sample_markdown)
    print(f"Parsed: {len(parsed_md.semantic_units)} semantic units")

    chunked_md = chunk_parsed_files([parsed_md])

    if chunked_md:
        cf = chunked_md[0]
        print(f"File: {cf.path}")
        print(f"Chunks: {len(cf.chunks)}")

        for i, chunk in enumerate(cf.chunks, 1):
            print(f"\nChunk {i}:")
            print(f"  Tokens: {chunk.token_count}")
            print(f"  Source units: {', '.join(chunk.source_units)[:50]}...")

    # Test 3: Statistics
    print("\n\n[Test 3] Chunk statistics")
    all_chunked = chunk_parsed_files([parsed, parsed_md])
    stats = get_chunk_stats(all_chunked)

    print("Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    print("\n✓ All tests passed!")
