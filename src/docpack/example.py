#!/usr/bin/env python3
"""
example.py
Docpack subsystem example – Demonstrates docpack generation

This example shows how to use the docpack subsystem to generate
a complete documentation bundle from pipeline results.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from docpack import generate_docpack, DocpackConfig


def example_basic():
    """
    Basic example showing docpack generation with default settings.

    Note: This example requires actual parsed files, chunks, embeddings,
    cluster results, and summary results from a real pipeline run.
    See src/main.py for the complete pipeline integration.
    """
    print("=" * 70)
    print("DOCPACK EXAMPLE - Basic Usage")
    print("=" * 70)

    print("\nThe docpack subsystem is integrated into the main pipeline.")
    print("Run the complete pipeline with:")
    print("  uv run python src/main.py <github_url_or_local_path>")
    print("\nExample:")
    print("  uv run python src/main.py https://github.com/xandwr/localdoc")
    print("\nThis will generate a .docpack.zip file in the output/ directory.")

    print("\n" + "=" * 70)
    print("DOCPACK STRUCTURE")
    print("=" * 70)
    print("""
The generated docpack contains:

docpack/
  metadata.json         - Pipeline configuration and statistics
  files.json           - Original file metadata
  chunks.json          - Chunk metadata with cluster assignments
  clusters.json        - Cluster information and assignments
  embeddings.npy       - NumPy array of embeddings

  text/
    files/
      0.txt            - Original normalized file text
      1.txt
      ...
    chunks/
      0.txt            - Individual chunk text
      1.txt
      ...

  summaries/
    cluster_0.short.md - Short cluster summary
    cluster_0.long.md  - Detailed cluster summary
    cluster_1.short.md
    cluster_1.long.md
    ...
    project_overview.md - Overall project summary
    """)


def example_custom_config():
    """
    Example showing custom docpack configuration.
    """
    print("=" * 70)
    print("DOCPACK EXAMPLE - Custom Configuration")
    print("=" * 70)

    print("\nCustomize docpack generation with CLI arguments:")
    print("")
    print("  --output-dir, -o        Output directory (default: output)")
    print("  --docpack-name          Name of docpack file (default: project.docpack)")
    print("  --no-zip                Skip ZIP compression")
    print("  --keep-uncompressed     Keep uncompressed directory after zipping")
    print("")
    print("Example:")
    print("  uv run python src/main.py https://github.com/owner/repo \\")
    print("    -o my_output \\")
    print("    --docpack-name myproject.docpack \\")
    print("    --keep-uncompressed")


if __name__ == "__main__":
    example_basic()
    print("\n")
    example_custom_config()
    print("\n")
