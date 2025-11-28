#!/usr/bin/env python3
"""
Example: Using Qwen-Agent for summarization in the doctown pipeline.

This demonstrates two approaches:
1. Direct replacement of SummaryModel with QwenAgentSummarizer
2. Integration into the main pipeline
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from summarize.qwen_agent_summarizer import QwenAgentSummarizer, QwenAgentConfig
from summarize.summarize import ClusterSummary, SummaryResult


# ============================================================
# Approach 1: Drop-in Replacement
# ============================================================

def summarize_cluster_with_agent(
    chunks: list,
    cluster_id: int,
    summarizer: QwenAgentSummarizer,
    max_chunks: int = 10,
) -> ClusterSummary:
    """
    Generate summaries for a cluster using qwen-agent.

    This is a drop-in replacement for the standard summarize_cluster function.

    Args:
        chunks: List of text chunks
        cluster_id: Cluster ID
        summarizer: QwenAgentSummarizer instance
        max_chunks: Max chunks to use

    Returns:
        ClusterSummary object
    """
    result = summarizer.summarize_cluster(
        chunks=chunks,
        cluster_id=cluster_id,
        max_chunks=max_chunks,
    )

    # Convert to standard ClusterSummary format
    return ClusterSummary(
        cluster_id=cluster_id,
        short_summary=result['short'],
        long_summary=result['long'],
        num_chunks=len(chunks),
        sample_chunks=chunks[:max_chunks],
        metadata={
            'summarizer': 'qwen-agent',
            'model': summarizer.config.model,
        }
    )


def summarize_all_clusters_with_agent(
    chunks: list,
    cluster_labels: np.ndarray,
    config: QwenAgentConfig = None,
    generate_project_summary: bool = True,
    max_chunks_per_cluster: int = 10,
    verbose: bool = True,
) -> SummaryResult:
    """
    Generate summaries for all clusters using qwen-agent.

    Drop-in replacement for summarize_all_clusters that uses qwen-agent backend.

    Args:
        chunks: List of Chunk objects
        cluster_labels: Cluster assignments
        config: QwenAgentConfig (optional)
        generate_project_summary: Generate project overview
        max_chunks_per_cluster: Max chunks per cluster
        verbose: Print progress

    Returns:
        SummaryResult object
    """
    if config is None:
        config = QwenAgentConfig(
            model="Qwen2.5-3B-Instruct",
            verbose=verbose,
        )

    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARIZATION (Qwen-Agent)")
        print("=" * 70)
        print(f"\nModel: {config.model}")

    # Initialize summarizer
    summarizer = QwenAgentSummarizer(config)

    # Group chunks by cluster
    cluster_chunks = {}
    for chunk, label in zip(chunks, cluster_labels):
        if label not in cluster_chunks:
            cluster_chunks[label] = []
        chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)
        cluster_chunks[label].append(chunk_text)

    # Remove noise cluster if present
    if -1 in cluster_chunks:
        if verbose:
            print(f"\nSkipping noise cluster (-1) with {len(cluster_chunks[-1])} chunks")
        del cluster_chunks[-1]

    n_clusters = len(cluster_chunks)

    if verbose:
        print(f"\nGenerating summaries for {n_clusters} clusters...")

    # Summarize each cluster
    cluster_summaries = []
    for cluster_id in sorted(cluster_chunks.keys()):
        summary = summarize_cluster_with_agent(
            chunks=cluster_chunks[cluster_id],
            cluster_id=cluster_id,
            summarizer=summarizer,
            max_chunks=max_chunks_per_cluster,
        )
        cluster_summaries.append(summary)

    # Generate project summary
    project_summary = None
    if generate_project_summary and cluster_summaries:
        if verbose:
            print(f"\nGenerating project overview...")

        cluster_summary_text = "\n\n".join(
            f"Cluster {cs.cluster_id} ({cs.num_chunks} chunks):\n{cs.long_summary}"
            for cs in cluster_summaries
        )

        project_prompt = f"""Based on the following cluster summaries from a codebase, provide a comprehensive project overview:

{cluster_summary_text}

Project Overview:"""

        project_summary = summarizer.generate_summary(project_prompt)

        if verbose:
            print(f"  ✓ Project summary: {project_summary[:150]}...")

    if verbose:
        print(f"\n✓ Summarization complete")

    return SummaryResult(
        cluster_summaries=cluster_summaries,
        project_summary=project_summary,
        model_name=config.model,
        total_clusters=n_clusters,
        metadata={
            'backend': 'qwen-agent',
            'max_chunks_per_cluster': max_chunks_per_cluster,
            'temperature': config.temperature,
        }
    )


# ============================================================
# Approach 2: Pipeline Integration
# ============================================================

def integrate_with_main_pipeline():
    """
    Example of how to integrate qwen-agent into main.py

    Replace the stage_summarize function in main.py with this:
    """
    example_code = '''
def stage_summarize(chunks, cluster_result: ClusterResult, config: PipelineConfig) -> SummaryResult:
    """
    Stage 6: Generate summaries using qwen-agent.
    """
    print("=" * 70)
    print("STAGE 6: SUMMARIZE (Qwen-Agent)")
    print("=" * 70)

    start_time = time.time()

    # Create qwen-agent config from pipeline config
    from summarize.qwen_agent_summarizer import QwenAgentConfig
    from summarize.example_qwen_agent import summarize_all_clusters_with_agent

    agent_config = QwenAgentConfig(
        model=config.summarize_model,
        temperature=0.2,
        verbose=True,
    )

    # Use qwen-agent backend
    summary_result = summarize_all_clusters_with_agent(
        chunks=chunks,
        cluster_labels=cluster_result.labels,
        config=agent_config,
        generate_project_summary=config.generate_project_summary,
        max_chunks_per_cluster=config.max_chunks_per_cluster,
    )

    elapsed = time.time() - start_time
    print(f"\\n✓ Summarization complete in {elapsed:.2f}s")

    return summary_result, elapsed
'''

    print("Integration Example:")
    print("=" * 70)
    print(example_code)
    print("=" * 70)


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    print("Qwen-Agent Integration Examples")
    print("=" * 70)

    # Show integration approach
    integrate_with_main_pipeline()

    print("\n\nTo use qwen-agent in the pipeline:")
    print("1. Update src/main.py stage_summarize() function")
    print("2. Or create a new --use-qwen-agent flag")
    print("3. Or set SUMMARIZE_BACKEND=qwen-agent environment variable")

    print("\n\nExample with mock data:")
    print("-" * 70)

    from dataclasses import dataclass

    @dataclass
    class MockChunk:
        text: str

    # Create test data
    test_chunks = [
        MockChunk("def embed_text(text): return model.encode(text)"),
        MockChunk("embeddings = embed_text(chunks)"),
        MockChunk("def cluster_embeddings(embeddings): return KMeans(n_clusters=5).fit(embeddings)"),
        MockChunk("clusters = cluster_embeddings(embeddings)"),
    ]

    test_labels = np.array([0, 0, 1, 1])

    # Create config
    config = QwenAgentConfig(
        model="Qwen2.5-3B-Instruct",
        # Uncomment if using local vLLM/Ollama:
        # model_server="http://localhost:8000/v1",
        temperature=0.2,
        verbose=True,
    )

    print("\nNote: This example requires a working LLM endpoint.")
    print("To test with local models, start vLLM/Ollama first:")
    print("  vllm serve Qwen/Qwen2.5-3B-Instruct --port 8000")
    print()

    # Uncomment to run:
    # result = summarize_all_clusters_with_agent(
    #     chunks=test_chunks,
    #     cluster_labels=test_labels,
    #     config=config,
    #     verbose=True,
    # )
    #
    # print("\nResults:")
    # for cs in result.cluster_summaries:
    #     print(f"\nCluster {cs.cluster_id}:")
    #     print(f"  Short: {cs.short_summary}")
    #     print(f"  Long: {cs.long_summary}")
