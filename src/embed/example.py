#!/usr/bin/env python3
# example.py
# Doctown v11 – Embed subsystem example

"""
Example usage of the embed subsystem.

This demonstrates:
1. Device detection and GPU acceleration
2. Embedding text chunks
3. Computing similarities
4. Full pipeline: parse → chunk → embed
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parse.parse import parse_file
from chunk.chunk import chunk_parsed_files, get_all_chunks
from embed.embed import (
    embed_chunks,
    get_device_info,
    create_embedding_model,
    find_most_similar,
    save_embeddings,
    load_embeddings,
)


def main():
    print("=" * 70)
    print("Doctown v11 - Embed Subsystem Example")
    print("=" * 70)

    # Example 1: Device information
    print("\n[Example 1] Device Detection")
    print("-" * 70)

    device_info = get_device_info()
    print("Device information:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")

    # Example 2: Simple text embedding
    print("\n\n[Example 2] Simple Text Embedding")
    print("-" * 70)

    simple_texts = [
        "Python is a high-level programming language.",
        "Machine learning models require training data.",
        "Embeddings convert text into numerical vectors.",
    ]

    print(f"Embedding {len(simple_texts)} texts...")
    simple_embeddings = embed_chunks(
        simple_texts,
        model_name="google/embeddinggemma-300m",
        show_progress=False,
    )

    print(f"\nResult shape: {simple_embeddings.shape}")
    print(f"Embedding dimension: {simple_embeddings.shape[1]}")
    print(f"Sample (first 5 dims): {simple_embeddings[0][:5]}")

    # Example 3: Full pipeline (parse → chunk → embed)
    print("\n\n[Example 3] Full Pipeline: Parse → Chunk → Embed")
    print("-" * 70)

    # Sample Python code
    python_code = b"""
import numpy as np
from typing import List

def calculate_mean(numbers: List[float]) -> float:
    \"\"\"Calculate the arithmetic mean of a list of numbers.\"\"\"
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)

def calculate_variance(numbers: List[float]) -> float:
    \"\"\"Calculate the variance of a list of numbers.\"\"\"
    if not numbers:
        return 0.0
    mean = calculate_mean(numbers)
    squared_diffs = [(x - mean) ** 2 for x in numbers]
    return sum(squared_diffs) / len(numbers)

class StatisticsCalculator:
    \"\"\"A class for calculating various statistics.\"\"\"

    def __init__(self, data: List[float]):
        self.data = data

    def mean(self) -> float:
        return calculate_mean(self.data)

    def variance(self) -> float:
        return calculate_variance(self.data)

    def std_dev(self) -> float:
        \"\"\"Calculate standard deviation.\"\"\"
        return self.variance() ** 0.5
"""

    # Sample Markdown documentation
    markdown_doc = b"""
# Statistics Module

## Overview
This module provides functions and classes for statistical calculations.

## Functions

### calculate_mean
Computes the arithmetic mean of a list of numbers.

**Parameters:**
- numbers: A list of float values

**Returns:**
- The mean value as a float

### calculate_variance
Computes the variance of a list of numbers.

## Classes

### StatisticsCalculator
A comprehensive class for statistical operations.

**Methods:**
- mean(): Returns the mean
- variance(): Returns the variance
- std_dev(): Returns the standard deviation
"""

    print("Step 1: Parse files")
    parsed_python = parse_file("statistics.py", python_code)
    parsed_markdown = parse_file("README.md", markdown_doc)
    print(f"  ✓ Parsed Python file: {len(parsed_python.semantic_units)} units")
    print(f"  ✓ Parsed Markdown file: {len(parsed_markdown.semantic_units)} units")

    print("\nStep 2: Chunk semantic units")
    chunked_files = chunk_parsed_files(
        [parsed_python, parsed_markdown],
        target_tokens=200,
    )
    all_chunks = get_all_chunks(chunked_files)
    print(f"  ✓ Created {len(all_chunks)} chunks")

    for i, chunk in enumerate(all_chunks, 1):
        print(f"    Chunk {i}: {chunk.token_count} tokens from {chunk.source_file}")

    print("\nStep 3: Embed chunks")
    chunk_embeddings = embed_chunks(
        all_chunks,  # Pass Chunk objects directly
        model_name="google/embeddinggemma-300m",
        show_progress=False,
    )
    print(f"  ✓ Generated embeddings shape: {chunk_embeddings.shape}")

    # Example 4: Similarity search
    print("\n\n[Example 4] Semantic Search with Embeddings")
    print("-" * 70)

    query = "How do I calculate the average of numbers?"
    print(f"Query: '{query}'")

    # Embed query
    print("\nEmbedding query...")
    model = create_embedding_model(model_name="google/embeddinggemma-300m")
    query_embedding = model.encode([query], show_progress=False)

    # Find most similar chunks
    indices, similarities = find_most_similar(
        chunk_embeddings,
        query_embedding[0],
        top_k=3,
    )

    print("\nTop 3 most similar chunks:")
    for rank, (idx, similarity) in enumerate(zip(indices, similarities), 1):
        chunk = all_chunks[idx]
        print(f"\n  Rank {rank} (similarity: {similarity:.4f})")
        print(f"  Source: {chunk.source_file}")
        print(f"  Units: {', '.join(chunk.source_units)}")
        preview = chunk.text[:100].replace('\n', ' ')
        print(f"  Preview: {preview}...")

    # Example 5: Save and load embeddings
    print("\n\n[Example 5] Saving and Loading Embeddings")
    print("-" * 70)

    save_path = "/tmp/doctown_embeddings.npy"
    print(f"Saving embeddings to {save_path}")
    save_embeddings(chunk_embeddings, save_path)

    print(f"\nLoading embeddings from {save_path}")
    loaded_embeddings = load_embeddings(save_path)

    # Verify
    if np.allclose(chunk_embeddings, loaded_embeddings):
        print("✓ Embeddings saved and loaded successfully!")
    else:
        print("✗ Error: Loaded embeddings don't match!")

    # Example 6: Compute all pairwise similarities
    print("\n\n[Example 6] Pairwise Chunk Similarities")
    print("-" * 70)

    # Compute similarity matrix (assuming normalized embeddings)
    similarity_matrix = chunk_embeddings @ chunk_embeddings.T

    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print("\nMost similar chunk pairs:")

    # Find top similar pairs (excluding self-similarity)
    np.fill_diagonal(similarity_matrix, -1)  # Ignore self-similarity
    flat_indices = np.argsort(similarity_matrix.flatten())[::-1]

    for i in range(min(3, len(flat_indices))):
        flat_idx = flat_indices[i]
        row = flat_idx // similarity_matrix.shape[1]
        col = flat_idx % similarity_matrix.shape[1]
        similarity = similarity_matrix[row, col]

        if similarity < 0:  # Skip if we've run out of valid pairs
            break

        chunk1 = all_chunks[row]
        chunk2 = all_chunks[col]

        print(f"\n  Pair {i+1} (similarity: {similarity:.4f})")
        print(f"    Chunk {row}: {chunk1.source_file} - {', '.join(chunk1.source_units[:2])}")
        print(f"    Chunk {col}: {chunk2.source_file} - {', '.join(chunk2.source_units[:2])}")

    print("\n" + "=" * 70)
    print("Example complete!")
    print("\n💡 Next steps:")
    print("  - Use these embeddings for clustering (cluster/ stage)")
    print("  - Generate summaries for each cluster (summarize/ stage)")
    print("  - Package everything into docpack (docpack/ stage)")
    print("=" * 70)


if __name__ == "__main__":
    main()
