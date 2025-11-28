#!/usr/bin/env python3
"""
main.py
Doctown v11 – Main orchestrator

Single binary entry point for the complete documentation synthesis pipeline.
Runs all subsystems sequentially: ingest → parse → chunk → embed → cluster → summarize → docpack

Usage:
    python main.py <github_url>
    python main.py <local_path>
    python main.py --url https://github.com/owner/repo
    python main.py --path /path/to/directory

Examples:
    python main.py https://github.com/xandwr/localdoc
    python main.py --url https://github.com/octocat/Hello-World --branch main
    python main.py --path ~/projects/myrepo
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import all subsystems
from ingest import ingest_github_repo, ingest_local_path, IngestError
from parse import parse_files, ParsedFile, ParseError
from chunk import chunk_parsed_files, get_all_chunks
from embed import embed_chunks
from cluster import cluster_embeddings, ClusterResult
from summarize import summarize_all_clusters, SummaryResult


# ============================================================
# Configuration
# ============================================================

@dataclass
class PipelineConfig:
    """Configuration for the pipeline."""
    # Ingest settings
    max_file_size: int = 1024 * 1024  # 1MB
    filter_binary: bool = True

    # Chunk settings
    min_tokens: int = 50
    max_tokens: int = 512
    target_tokens: int = 300

    # Embed settings
    embed_model: str = "google/embeddinggemma-300m"
    embed_batch_size: int = 32
    embed_normalize: bool = True

    # Cluster settings
    cluster_algorithm: str = "kmeans"
    n_clusters: int = 5
    auto_find_k: bool = True  # Auto-find optimal k
    k_range_min: int = 2
    k_range_max: int = 10

    # Summarize settings
    summarize_model: str = "Qwen/Qwen3-1.7B"
    generate_project_summary: bool = True
    max_chunks_per_cluster: int = 10
    summarize_load_in_8bit: bool = False


# ============================================================
# Pipeline Statistics
# ============================================================

@dataclass
class PipelineStats:
    """Statistics collected during pipeline execution."""
    # Stage timings
    ingest_time: float = 0.0
    parse_time: float = 0.0
    chunk_time: float = 0.0
    embed_time: float = 0.0
    cluster_time: float = 0.0
    summarize_time: float = 0.0
    total_time: float = 0.0

    # Stage outputs
    files_ingested: int = 0
    files_parsed: int = 0
    chunks_created: int = 0
    embeddings_dimension: int = 0
    clusters_found: int = 0
    summaries_generated: int = 0

    # Additional metrics
    total_bytes: int = 0
    total_lines: int = 0
    cluster_silhouette: Optional[float] = None

    def print_summary(self):
        """Print a formatted summary of pipeline statistics."""
        print("\n" + "=" * 70)
        print("PIPELINE SUMMARY")
        print("=" * 70)

        print("\nStage Timings:")
        print(f"  Ingest:     {self.ingest_time:>8.2f}s  ({self.files_ingested} files, {self.total_bytes/1024/1024:.1f} MB)")
        print(f"  Parse:      {self.parse_time:>8.2f}s  ({self.files_parsed} files, {self.total_lines} lines)")
        print(f"  Chunk:      {self.chunk_time:>8.2f}s  ({self.chunks_created} chunks)")
        print(f"  Embed:      {self.embed_time:>8.2f}s  ({self.embeddings_dimension}D embeddings)")
        print(f"  Cluster:    {self.cluster_time:>8.2f}s  ({self.clusters_found} clusters)")
        print(f"  Summarize:  {self.summarize_time:>8.2f}s  ({self.summaries_generated} summaries)")
        print(f"  {'─' * 40}")
        print(f"  Total:      {self.total_time:>8.2f}s")

        if self.cluster_silhouette is not None:
            print(f"\nCluster Quality:")
            print(f"  Silhouette Score: {self.cluster_silhouette:.3f} (range: -1 to 1, higher is better)")

        print("\nPipeline Status: ✓ Complete")
        print("=" * 70)


# ============================================================
# URL Parsing
# ============================================================

def parse_github_url(url: str) -> tuple[str, str, str]:
    """
    Parse a GitHub URL into owner, repo, and branch.

    Supports formats:
    - https://github.com/owner/repo
    - https://github.com/owner/repo/tree/branch
    - github.com/owner/repo

    Args:
        url: GitHub URL

    Returns:
        Tuple of (owner, repo, branch)
    """
    # Remove protocol if present
    url = url.replace("https://", "").replace("http://", "")

    # Remove github.com prefix
    if url.startswith("github.com/"):
        url = url[11:]

    # Split path
    parts = url.split("/")

    if len(parts) < 2:
        raise ValueError(f"Invalid GitHub URL: {url}")

    owner = parts[0]
    repo = parts[1]

    # Check for branch in URL (e.g., /tree/branch_name)
    branch = "main"  # default
    if len(parts) >= 4 and parts[2] == "tree":
        branch = parts[3]

    return owner, repo, branch


# ============================================================
# Pipeline Stages
# ============================================================

def stage_ingest(url: Optional[str], path: Optional[str], config: PipelineConfig) -> list:
    """
    Stage 1: Ingest files from GitHub or local path.

    Args:
        url: GitHub URL (if provided)
        path: Local path (if provided)
        config: Pipeline configuration

    Returns:
        List of FileEntry objects
    """
    print("\n" + "=" * 70)
    print("STAGE 1: INGEST")
    print("=" * 70)

    start_time = time.time()

    if url:
        # Parse GitHub URL
        owner, repo, branch = parse_github_url(url)
        print(f"\nIngesting GitHub repository:")
        print(f"  Owner:  {owner}")
        print(f"  Repo:   {repo}")
        print(f"  Branch: {branch}")

        files = ingest_github_repo(
            owner=owner,
            repo=repo,
            branch=branch,
            max_file_size=config.max_file_size,
            filter_binary=config.filter_binary,
        )
    elif path:
        print(f"\nIngesting local path: {path}")
        files = ingest_local_path(
            path,
            max_file_size=config.max_file_size,
            filter_binary=config.filter_binary,
        )
    else:
        raise ValueError("Either url or path must be provided")

    elapsed = time.time() - start_time

    total_bytes = sum(len(f.content) for f in files)
    print(f"\n✓ Ingested {len(files)} files ({total_bytes/1024/1024:.1f} MB) in {elapsed:.2f}s")

    return files, elapsed, total_bytes


def stage_parse(file_entries: list, config: PipelineConfig) -> list[ParsedFile]:
    """
    Stage 2: Parse files into structured format.

    Args:
        file_entries: List of FileEntry objects from ingest
        config: Pipeline configuration

    Returns:
        List of ParsedFile objects
    """
    print("\n" + "=" * 70)
    print("STAGE 2: PARSE")
    print("=" * 70)

    start_time = time.time()

    print(f"\nParsing {len(file_entries)} files...")
    parsed_files = parse_files(file_entries)

    elapsed = time.time() - start_time

    # Calculate statistics
    total_lines = sum(pf.line_count for pf in parsed_files)
    total_units = sum(len(pf.semantic_units) for pf in parsed_files)

    print(f"\n✓ Parsed {len(parsed_files)} files in {elapsed:.2f}s")
    print(f"  Total lines: {total_lines:,}")
    print(f"  Semantic units: {total_units:,}")

    # Show file type distribution
    type_counts = {}
    for pf in parsed_files:
        type_counts[pf.file_type.value] = type_counts.get(pf.file_type.value, 0) + 1

    print(f"\n  File types:")
    for file_type, count in sorted(type_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"    {file_type}: {count}")

    return parsed_files, elapsed, total_lines


def stage_chunk(parsed_files: list[ParsedFile], config: PipelineConfig):
    """
    Stage 3: Chunk parsed files into embedding-ready chunks.

    Args:
        parsed_files: List of ParsedFile objects
        config: Pipeline configuration

    Returns:
        List of Chunk objects
    """
    print("\n" + "=" * 70)
    print("STAGE 3: CHUNK")
    print("=" * 70)

    start_time = time.time()

    print(f"\nChunking {len(parsed_files)} files...")
    print(f"  Target tokens: {config.target_tokens}")
    print(f"  Range: {config.min_tokens}-{config.max_tokens}")

    chunked_files = chunk_parsed_files(
        parsed_files,
        min_tokens=config.min_tokens,
        max_tokens=config.max_tokens,
        target_tokens=config.target_tokens,
    )

    # Flatten chunks
    chunks = get_all_chunks(chunked_files)

    elapsed = time.time() - start_time

    # Calculate statistics
    total_tokens = sum(c.token_count for c in chunks)
    avg_tokens = total_tokens / len(chunks) if chunks else 0

    print(f"\n✓ Created {len(chunks)} chunks in {elapsed:.2f}s")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Average tokens/chunk: {avg_tokens:.0f}")

    return chunks, elapsed


def stage_embed(chunks, config: PipelineConfig):
    """
    Stage 4: Generate embeddings for chunks.

    Args:
        chunks: List of Chunk objects
        config: Pipeline configuration

    Returns:
        numpy array of embeddings
    """
    print("\n" + "=" * 70)
    print("STAGE 4: EMBED")
    print("=" * 70)

    start_time = time.time()

    print(f"\nEmbedding {len(chunks)} chunks...")
    print(f"  Model: {config.embed_model}")
    print(f"  Batch size: {config.embed_batch_size}")

    embeddings = embed_chunks(
        chunks,
        model_name=config.embed_model,
        batch_size=config.embed_batch_size,
        normalize=config.embed_normalize,
        show_progress=True,
    )

    elapsed = time.time() - start_time

    print(f"\n✓ Generated embeddings in {elapsed:.2f}s")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Dimension: {embeddings.shape[1]}")

    return embeddings, elapsed


def stage_cluster(embeddings, config: PipelineConfig) -> ClusterResult:
    """
    Stage 5: Cluster embeddings into topics.

    Args:
        embeddings: numpy array of embeddings
        config: Pipeline configuration

    Returns:
        ClusterResult object
    """
    print("\n" + "=" * 70)
    print("STAGE 5: CLUSTER")
    print("=" * 70)

    start_time = time.time()

    n_clusters = config.n_clusters

    # Auto-find optimal k if enabled
    if config.auto_find_k and len(embeddings) >= config.k_range_min:
        from cluster import find_optimal_k

        k_max = min(config.k_range_max, len(embeddings) - 1)
        k_range = range(config.k_range_min, k_max + 1)

        print(f"\nFinding optimal number of clusters...")
        print(f"  Testing k in range: {list(k_range)}")

        optimal_k, scores = find_optimal_k(
            embeddings,
            k_range=k_range,
            metric="silhouette",
            verbose=True,
        )

        n_clusters = optimal_k
        print(f"\n✓ Using optimal k = {n_clusters}")

    print(f"\nClustering {len(embeddings)} embeddings...")
    print(f"  Algorithm: {config.cluster_algorithm}")
    print(f"  Clusters: {n_clusters}")

    result = cluster_embeddings(
        embeddings,
        algorithm=config.cluster_algorithm,
        n_clusters=n_clusters,
        verbose=True,
    )

    elapsed = time.time() - start_time

    print(f"\n✓ Clustering complete in {elapsed:.2f}s")

    # Show cluster distribution
    cluster_sizes = result.get_cluster_sizes()
    print(f"\n  Cluster distribution:")
    for cluster_id in sorted(cluster_sizes.keys()):
        size = cluster_sizes[cluster_id]
        pct = 100 * size / len(embeddings)
        print(f"    Cluster {cluster_id}: {size:4d} items ({pct:5.1f}%)")

    return result, elapsed


def stage_summarize(chunks, cluster_result: ClusterResult, config: PipelineConfig) -> SummaryResult:
    """
    Stage 6: Generate summaries for clusters.

    Args:
        chunks: List of Chunk objects
        cluster_result: ClusterResult from clustering stage
        config: Pipeline configuration

    Returns:
        SummaryResult object
    """
    print("\n" + "=" * 70)
    print("STAGE 6: SUMMARIZE")
    print("=" * 70)

    start_time = time.time()

    print(f"\nGenerating summaries for {cluster_result.n_clusters} clusters...")
    print(f"  Model: {config.summarize_model}")
    print(f"  Max chunks per cluster: {config.max_chunks_per_cluster}")
    print(f"  Project summary: {config.generate_project_summary}")

    summary_result = summarize_all_clusters(
        chunks=chunks,
        cluster_labels=cluster_result.labels,
        model_name=config.summarize_model,
        generate_project_summary=config.generate_project_summary,
        max_chunks_per_cluster=config.max_chunks_per_cluster,
        load_in_8bit=config.summarize_load_in_8bit,
        verbose=True,
    )

    elapsed = time.time() - start_time

    print(f"\n✓ Summarization complete in {elapsed:.2f}s")
    print(f"  Generated {len(summary_result.cluster_summaries)} cluster summaries")
    if summary_result.project_summary:
        print(f"  Generated project overview")

    return summary_result, elapsed


# ============================================================
# Main Pipeline
# ============================================================

def run_pipeline(
    url: Optional[str] = None,
    path: Optional[str] = None,
    config: Optional[PipelineConfig] = None,
) -> PipelineStats:
    """
    Run the complete documentation synthesis pipeline.

    Args:
        url: GitHub URL to ingest (optional)
        path: Local path to ingest (optional)
        config: Pipeline configuration (optional, uses defaults if not provided)

    Returns:
        PipelineStats object with execution statistics
    """
    if config is None:
        config = PipelineConfig()

    stats = PipelineStats()
    pipeline_start = time.time()

    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "DOCTOWN v11 PIPELINE" + " " * 28 + "║")
    print("╚" + "═" * 68 + "╝")

    try:
        # Stage 1: Ingest
        file_entries, ingest_time, total_bytes = stage_ingest(url, path, config)
        stats.ingest_time = ingest_time
        stats.files_ingested = len(file_entries)
        stats.total_bytes = total_bytes

        # Stage 2: Parse
        parsed_files, parse_time, total_lines = stage_parse(file_entries, config)
        stats.parse_time = parse_time
        stats.files_parsed = len(parsed_files)
        stats.total_lines = total_lines

        # Stage 3: Chunk
        chunks, chunk_time = stage_chunk(parsed_files, config)
        stats.chunk_time = chunk_time
        stats.chunks_created = len(chunks)

        # Stage 4: Embed
        embeddings, embed_time = stage_embed(chunks, config)
        stats.embed_time = embed_time
        stats.embeddings_dimension = embeddings.shape[1] if len(embeddings) > 0 else 0

        # Stage 5: Cluster
        cluster_result, cluster_time = stage_cluster(embeddings, config)
        stats.cluster_time = cluster_time
        stats.clusters_found = cluster_result.n_clusters
        stats.cluster_silhouette = cluster_result.silhouette

        # Stage 6: Summarize
        summary_result, summarize_time = stage_summarize(chunks, cluster_result, config)
        stats.summarize_time = summarize_time
        stats.summaries_generated = len(summary_result.cluster_summaries)

        # Calculate total time
        stats.total_time = time.time() - pipeline_start

        # Print summary
        stats.print_summary()

        return stats

    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ============================================================
# CLI Interface
# ============================================================

def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Doctown v11 - Documentation synthesis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s https://github.com/xandwr/localdoc
  %(prog)s --url https://github.com/octocat/Hello-World --branch main
  %(prog)s --path ~/projects/myrepo
  %(prog)s https://github.com/owner/repo --clusters 8
        """
    )

    # Input source (positional or flags)
    parser.add_argument(
        "source",
        nargs="?",
        help="GitHub URL or local path to ingest"
    )
    parser.add_argument(
        "--url",
        help="GitHub repository URL"
    )
    parser.add_argument(
        "--path",
        help="Local directory path"
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="Git branch to use (default: main)"
    )

    # Pipeline configuration
    parser.add_argument(
        "--clusters", "-k",
        type=int,
        default=5,
        help="Number of clusters (default: 5)"
    )
    parser.add_argument(
        "--auto-k",
        action="store_true",
        default=True,
        help="Automatically find optimal number of clusters (default: True)"
    )
    parser.add_argument(
        "--no-auto-k",
        action="store_false",
        dest="auto_k",
        help="Disable automatic cluster count detection"
    )
    parser.add_argument(
        "--cluster-algorithm",
        choices=["kmeans", "hdbscan"],
        default="kmeans",
        help="Clustering algorithm (default: kmeans)"
    )
    parser.add_argument(
        "--embed-model",
        default="google/embeddinggemma-300m",
        help="Embedding model (default: google/embeddinggemma-300m)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens per chunk (default: 512)"
    )
    parser.add_argument(
        "--summarize-model",
        default="Qwen/Qwen3-1.7B",
        help="Summarization model (default: Qwen/Qwen3-1.7B)"
    )
    parser.add_argument(
        "--no-project-summary",
        action="store_false",
        dest="project_summary",
        help="Skip generating project-level summary"
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load summarization model in 8-bit to save memory"
    )

    args = parser.parse_args()

    # Determine source
    url = None
    path = None

    if args.source:
        # Auto-detect if it's a URL or path
        if "github.com" in args.source or args.source.startswith("http"):
            url = args.source
        else:
            path = args.source

    if args.url:
        url = args.url
    if args.path:
        path = args.path

    if not url and not path:
        parser.print_help()
        print("\nError: Please provide a GitHub URL or local path")
        sys.exit(1)

    # Build configuration
    config = PipelineConfig(
        n_clusters=args.clusters,
        auto_find_k=args.auto_k,
        cluster_algorithm=args.cluster_algorithm,
        embed_model=args.embed_model,
        max_tokens=args.max_tokens,
        summarize_model=args.summarize_model,
        generate_project_summary=args.project_summary,
        summarize_load_in_8bit=args.load_in_8bit,
    )

    # Run pipeline
    run_pipeline(url=url, path=path, config=config)


if __name__ == "__main__":
    main()
