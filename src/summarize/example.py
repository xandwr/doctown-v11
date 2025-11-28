#!/usr/bin/env python3
"""
example.py
Doctown v11 – Summarize subsystem standalone example

Demonstrates how to use the summarize subsystem with mock data.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from summarize import summarize_all_clusters, print_summary_report, save_summary_result


# ============================================================
# Mock Data Classes
# ============================================================

class MockChunk:
    """Mock Chunk object for testing."""
    def __init__(self, text: str):
        self.text = text


# ============================================================
# Example
# ============================================================

def main():
    print("=" * 70)
    print("SUMMARIZE SUBSYSTEM - STANDALONE EXAMPLE")
    print("=" * 70)

    # Create mock chunks representing a simple Python project
    print("\n[1] Creating mock chunks...")
    print("-" * 70)

    chunks = [
        # Cluster 0: Math operations
        MockChunk("def add(a, b):\n    return a + b"),
        MockChunk("def subtract(a, b):\n    return a - b"),
        MockChunk("def multiply(a, b):\n    return a * b"),
        MockChunk("class Calculator:\n    def __init__(self):\n        self.result = 0"),

        # Cluster 1: File I/O
        MockChunk("def read_file(path):\n    with open(path, 'r') as f:\n        return f.read()"),
        MockChunk("def write_file(path, content):\n    with open(path, 'w') as f:\n        f.write(content)"),
        MockChunk("import json\ndef load_json(path):\n    with open(path) as f:\n        return json.load(f)"),

        # Cluster 2: Data processing
        MockChunk("import pandas as pd\ndef load_csv(path):\n    return pd.read_csv(path)"),
        MockChunk("def filter_data(df, column, value):\n    return df[df[column] == value]"),
        MockChunk("def aggregate_data(df, group_by, agg_func):\n    return df.groupby(group_by).agg(agg_func)"),
    ]

    # Create cluster labels
    cluster_labels = np.array([
        0, 0, 0, 0,  # Math operations
        1, 1, 1,     # File I/O
        2, 2, 2,     # Data processing
    ])

    print(f"✓ Created {len(chunks)} mock chunks")
    print(f"✓ Organized into {len(set(cluster_labels))} clusters")

    # Show cluster distribution
    print("\nCluster distribution:")
    for cluster_id in sorted(set(cluster_labels)):
        count = np.sum(cluster_labels == cluster_id)
        print(f"  Cluster {cluster_id}: {count} chunks")

    # Generate summaries
    print("\n\n[2] Generating summaries...")
    print("-" * 70)

    result = summarize_all_clusters(
        chunks=chunks,
        cluster_labels=cluster_labels,
        generate_project_summary=True,
        max_chunks_per_cluster=5,
        verbose=True,
    )

    # Print report
    print("\n\n[3] Summary Report")
    print("-" * 70)

    print_summary_report(result)

    # Save to file
    print("\n\n[4] Saving summary to file...")
    print("-" * 70)

    output_path = "/tmp/doctown_summary_example.txt"
    save_summary_result(result, output_path)

    print(f"\n✓ Summary saved to: {output_path}")
    print(f"  You can view it with: cat {output_path}")

    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
