#!/usr/bin/env python3
# example.py
# Doctown v11 – Chunk subsystem example

"""
Example usage of the chunk subsystem.

This demonstrates:
1. Loading parsed files
2. Chunking semantic units
3. Inspecting chunk statistics
4. Preparing chunks for embedding
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parse.parse import parse_file
from chunk.chunk import (
    chunk_parsed_files,
    get_all_chunks,
    get_chunk_stats,
    estimate_token_count,
)


def main():
    print("=" * 60)
    print("Doctown v11 - Chunk Subsystem Example")
    print("=" * 60)

    # Example 1: Chunk a Python file
    print("\n[Example 1] Chunking Python code")
    print("-" * 60)

    python_code = b"""
import os
import sys
from typing import List, Optional

def process_data(items: List[str]) -> List[str]:
    \"\"\"Process a list of items.\"\"\"
    return [item.strip().lower() for item in items]

class DataProcessor:
    \"\"\"A class for processing data.\"\"\"

    def __init__(self, config: dict):
        self.config = config
        self.results = []

    def process(self, data: List[str]) -> None:
        \"\"\"Process data and store results.\"\"\"
        for item in data:
            processed = self._transform(item)
            self.results.append(processed)

    def _transform(self, item: str) -> str:
        \"\"\"Transform a single item.\"\"\"
        return item.upper()

    def get_results(self) -> List[str]:
        \"\"\"Get processing results.\"\"\"
        return self.results
"""

    parsed_python = parse_file("processor.py", python_code)
    print(f"File: processor.py")
    print(f"Semantic units found: {len(parsed_python.semantic_units)}")

    for unit in parsed_python.semantic_units:
        tokens = estimate_token_count(unit.content)
        print(f"  - {unit.type.value}: {unit.name} ({tokens} tokens)")

    chunked = chunk_parsed_files([parsed_python], target_tokens=150)

    if chunked:
        cf = chunked[0]
        print(f"\nChunks created: {len(cf.chunks)}")

        for i, chunk in enumerate(cf.chunks, 1):
            print(f"\n  Chunk {i}:")
            print(f"    Tokens: {chunk.token_count}")
            print(f"    Source units: {', '.join(chunk.source_units)}")
            if chunk.metadata.get("merged"):
                print(f"    Merged {chunk.metadata.get('unit_count', 0)} units")
            if chunk.metadata.get("split"):
                print(f"    Split part {chunk.metadata.get('part')}/{chunk.metadata.get('total_parts')}")

    # Example 2: Chunk a Markdown document
    print("\n\n[Example 2] Chunking Markdown documentation")
    print("-" * 60)

    markdown_doc = b"""
# Project Documentation

## Overview

This project provides a comprehensive solution for data processing.
It includes multiple components working together seamlessly.

## Installation

To install the project, run:

```bash
pip install our-project
```

## Quick Start

Here's a simple example:

```python
from our_project import DataProcessor

processor = DataProcessor(config={})
processor.process(["item1", "item2"])
results = processor.get_results()
```

## Advanced Usage

### Configuration Options

You can customize the processor with these options:

- `max_workers`: Number of parallel workers (default: 4)
- `timeout`: Processing timeout in seconds (default: 30)
- `verbose`: Enable verbose logging (default: False)

### Error Handling

The processor handles errors gracefully. When an error occurs,
it logs the issue and continues processing remaining items.

## API Reference

### DataProcessor

The main class for data processing.

**Methods:**

- `process(data)`: Process a list of items
- `get_results()`: Retrieve processing results
- `reset()`: Clear internal state

## Contributing

We welcome contributions! Please see CONTRIBUTING.md for details.

## License

MIT License - see LICENSE file for details.
"""

    parsed_md = parse_file("README.md", markdown_doc)
    print(f"File: README.md")
    print(f"Semantic units found: {len(parsed_md.semantic_units)}")

    chunked_md = chunk_parsed_files([parsed_md], target_tokens=200)

    if chunked_md:
        cf = chunked_md[0]
        print(f"\nChunks created: {len(cf.chunks)}")

        for i, chunk in enumerate(cf.chunks, 1):
            print(f"\n  Chunk {i}: {chunk.token_count} tokens")
            # Show first line of chunk for context
            first_line = chunk.text.split('\n')[0][:60]
            print(f"    Starts with: {first_line}...")

    # Example 3: Statistics across multiple files
    print("\n\n[Example 3] Statistics for multiple files")
    print("-" * 60)

    all_chunked = chunk_parsed_files([parsed_python, parsed_md])
    stats = get_chunk_stats(all_chunked)

    print(f"Files processed: {stats['total_files']}")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Average tokens/chunk: {stats['avg_tokens_per_chunk']:.1f}")
    print(f"Token range: {stats['min_tokens']} - {stats['max_tokens']}")
    print(f"Merged chunks: {stats['merged_chunks']}")
    print(f"Split chunks: {stats['split_chunks']}")

    # Example 4: Prepare for embedding
    print("\n\n[Example 4] Preparing chunks for embedding stage")
    print("-" * 60)

    all_chunks = get_all_chunks(all_chunked)
    print(f"Total chunks ready for embedding: {len(all_chunks)}")

    # Show what would be sent to embedding model
    print("\nSample chunks (first 3):")
    for i, chunk in enumerate(all_chunks[:3], 1):
        print(f"\n  Chunk {i} (from {chunk.source_file}):")
        print(f"    Tokens: {chunk.token_count}")
        preview = chunk.text[:100].replace('\n', ' ')
        print(f"    Text: {preview}...")

    # This list of chunk.text strings would be passed to embed/
    chunk_texts = [chunk.text for chunk in all_chunks]
    print(f"\n✓ Ready to pass {len(chunk_texts)} text strings to embed/ stage")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
