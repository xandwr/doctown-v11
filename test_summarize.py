#!/usr/bin/env python3
"""
Quick test to verify the summarization fix works.

This will test with synthetic data to ensure summaries are generated correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from summarize.summarize import SummaryModel, summarize_cluster
from dataclasses import dataclass

@dataclass
class MockChunk:
    """Mock chunk for testing."""
    text: str

def test_summary_generation():
    """Test that summaries are generated (not empty)."""
    print("=" * 70)
    print("TESTING SUMMARIZATION FIX")
    print("=" * 70)

    # Create test data
    test_chunks = [
        MockChunk("def calculate_sum(a, b): return a + b"),
        MockChunk("def calculate_product(a, b): return a * b"),
        MockChunk("class MathOperations: pass"),
    ]

    print("\n1. Loading model...")
    print(f"   Model: Qwen/Qwen3-4B")

    try:
        model = SummaryModel(
            model_name="Qwen/Qwen3-4B",
            device="cpu",  # Use CPU for testing
        )
        print("   ✓ Model loaded")
    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        print("\n   Note: If model not found, it will be downloaded automatically.")
        print("   This may take a few minutes on first run.")
        return

    print("\n2. Generating cluster summary...")

    try:
        result = summarize_cluster(
            chunks=[c.text for c in test_chunks],
            cluster_id=0,
            model=model,
            max_chunks=3,
            verbose=True,
        )
        print("   ✓ Summary generated")
    except Exception as e:
        print(f"   ✗ Failed to generate summary: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nShort Summary:")
    print(f"  Length: {len(result.short_summary)} characters")
    print(f"  Content: {result.short_summary[:200]}")

    print(f"\nLong Summary:")
    print(f"  Length: {len(result.long_summary)} characters")
    print(f"  Content: {result.long_summary[:200]}")

    # Verify summaries are not empty
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    issues = []

    if not result.short_summary or len(result.short_summary.strip()) == 0:
        issues.append("Short summary is empty")
    else:
        print("✓ Short summary has content")

    if not result.long_summary or len(result.long_summary.strip()) == 0:
        issues.append("Long summary is empty")
    else:
        print("✓ Long summary has content")

    # Check for garbage output (repeating prompt)
    if "Summary:" in result.short_summary:
        issues.append("Short summary contains prompt template")
    else:
        print("✓ Short summary does not contain prompt")

    if "Detailed Summary:" in result.long_summary:
        issues.append("Long summary contains prompt template")
    else:
        print("✓ Long summary does not contain prompt")

    print("\n" + "=" * 70)
    if issues:
        print("✗ TEST FAILED")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ TEST PASSED - Summaries are working correctly!")
    print("=" * 70)

if __name__ == "__main__":
    test_summary_generation()
