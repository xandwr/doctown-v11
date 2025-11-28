# docpack.py
# Doctown v11 – Docpack subsystem: Assemble final documentation bundle

import json
import shutil
import zipfile
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from parse.parse import ParsedFile, FileType
from chunk.chunk import Chunk, ChunkedFile
from cluster.cluster import ClusterResult
from summarize.summarize import SummaryResult


# ============================================================
# Exceptions
# ============================================================

class DocpackError(Exception):
    """Error during docpack generation."""
    pass


# ============================================================
# Configuration
# ============================================================

@dataclass
class DocpackConfig:
    """Configuration for docpack generation."""
    output_dir: str = "output"
    docpack_name: str = "project.docpack"

    # Metadata
    source_type: str = "unknown"  # "github", "local", etc.
    source_owner: Optional[str] = None
    source_repo: Optional[str] = None
    source_branch: Optional[str] = None
    source_commit: Optional[str] = None
    source_path: Optional[str] = None

    # Pipeline config to include
    embed_model: str = "google/embeddinggemma-300m"
    summarizer_model: str = "Qwen/Qwen3-1.7B"
    chunk_target_tokens: int = 300
    chunk_min_tokens: int = 50
    chunk_max_tokens: int = 512

    # Options
    compress_to_zip: bool = True
    keep_uncompressed: bool = False


# ============================================================
# Output Format
# ============================================================

@dataclass
class DocpackResult:
    """Result of docpack generation."""
    docpack_path: str
    docpack_dir: str
    total_size_bytes: int
    files_count: int
    chunks_count: int
    clusters_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Metadata Generation
# ============================================================

def generate_metadata(
    config: DocpackConfig,
    parsed_files: List[ParsedFile],
    chunks: List[Chunk],
    embeddings: np.ndarray,
    cluster_result: ClusterResult,
) -> dict:
    """
    Generate metadata.json content.

    Args:
        config: Docpack configuration
        parsed_files: List of parsed files
        chunks: List of chunks
        embeddings: Embeddings array
        cluster_result: Clustering result

    Returns:
        Dictionary containing metadata
    """
    # Calculate stats
    total_lines = sum(pf.line_count for pf in parsed_files)
    total_semantic_units = sum(len(pf.semantic_units) for pf in parsed_files)

    metadata = {
        "docpack_version": "11",
        "generator": "doctown-v11",
        "generated_at": datetime.now(timezone.utc).isoformat(),

        "source": {},

        "stats": {
            "files": len(parsed_files),
            "lines": total_lines,
            "semantic_units": total_semantic_units,
            "chunks": len(chunks),
            "embedding_dim": embeddings.shape[1] if len(embeddings) > 0 else 0,
            "clusters": cluster_result.n_clusters,
        },

        "pipeline": {
            "embed_model": config.embed_model,
            "summarizer_model": config.summarizer_model,
            "chunk_target_tokens": config.chunk_target_tokens,
            "chunk_min_tokens": config.chunk_min_tokens,
            "chunk_max_tokens": config.chunk_max_tokens,
        }
    }

    # Add source information
    if config.source_type == "github":
        metadata["source"] = {
            "type": "github",
            "owner": config.source_owner,
            "repo": config.source_repo,
            "branch": config.source_branch,
            "commit": config.source_commit,
        }
    elif config.source_type == "local":
        metadata["source"] = {
            "type": "local",
            "path": config.source_path,
        }
    else:
        metadata["source"] = {
            "type": config.source_type,
        }

    return metadata


def generate_files_json(parsed_files: List[ParsedFile]) -> list:
    """
    Generate files.json content.

    Args:
        parsed_files: List of parsed files

    Returns:
        List of file metadata dictionaries
    """
    files = []
    for i, pf in enumerate(parsed_files):
        files.append({
            "id": i,
            "path": pf.path,
            "file_type": pf.file_type.value,
            "language": pf.language,
            "size_bytes": pf.size_bytes,
            "line_count": pf.line_count,
            "semantic_units": len(pf.semantic_units),
        })
    return files


def generate_chunks_json(
    chunks: List[Chunk],
    parsed_files: List[ParsedFile],
    cluster_result: ClusterResult,
) -> list:
    """
    Generate chunks.json content.

    Args:
        chunks: List of chunks
        parsed_files: List of parsed files (to map file paths to IDs)
        cluster_result: Clustering result (for cluster assignments)

    Returns:
        List of chunk metadata dictionaries
    """
    # Build file path to ID mapping
    file_path_to_id = {pf.path: i for i, pf in enumerate(parsed_files)}

    chunks_list = []
    for i, chunk in enumerate(chunks):
        file_id = file_path_to_id.get(chunk.source_file, 0)
        cluster_id = int(cluster_result.labels[i])

        # Extract line range from chunk metadata if available
        start_line = chunk.metadata.get("start_line", 0)
        end_line = chunk.metadata.get("end_line", 0)

        chunks_list.append({
            "id": i,
            "file_id": file_id,
            "cluster_id": cluster_id,
            "start_line": start_line,
            "end_line": end_line,
            "token_count": chunk.token_count,
            "unit_ids": chunk.source_units if isinstance(chunk.source_units, list) else [],
            "text_path": f"text/chunks/{i}.txt",
        })

    return chunks_list


def generate_clusters_json(cluster_result: ClusterResult) -> dict:
    """
    Generate clusters.json content.

    Args:
        cluster_result: Clustering result

    Returns:
        Dictionary containing cluster metadata
    """
    cluster_sizes = cluster_result.get_cluster_sizes()

    clusters_list = []
    for cluster_id in sorted(cluster_sizes.keys()):
        if cluster_id == -1:  # Skip noise points (HDBSCAN)
            continue

        chunk_ids = cluster_result.get_cluster_members(cluster_id).tolist()

        clusters_list.append({
            "id": cluster_id,
            "chunk_ids": chunk_ids,
            "size": len(chunk_ids),
        })

    clusters_dict = {
        "algorithm": cluster_result.algorithm,
        "k": cluster_result.n_clusters,
        "silhouette_score": float(cluster_result.silhouette) if cluster_result.silhouette else None,
        "clusters": clusters_list,
    }

    return clusters_dict


# ============================================================
# File Writing
# ============================================================

def write_json_file(path: Path, data: Any):
    """Write data to JSON file with pretty formatting."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_text_files(
    base_dir: Path,
    parsed_files: List[ParsedFile],
    chunks: List[Chunk],
):
    """
    Write text files for both original files and chunks.

    Args:
        base_dir: Base directory for text files
        parsed_files: List of parsed files
        chunks: List of chunks
    """
    # Create directories
    files_dir = base_dir / "files"
    chunks_dir = base_dir / "chunks"
    files_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Write original file texts
    for i, pf in enumerate(parsed_files):
        file_path = files_dir / f"{i}.txt"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(pf.normalized_text)

    # Write chunk texts
    for i, chunk in enumerate(chunks):
        chunk_path = chunks_dir / f"{i}.txt"
        with open(chunk_path, 'w', encoding='utf-8') as f:
            f.write(chunk.text)


def write_summary_files(
    base_dir: Path,
    summary_result: SummaryResult,
):
    """
    Write summary markdown files.

    Args:
        base_dir: Base directory for summaries
        summary_result: Summarization result
    """
    base_dir.mkdir(parents=True, exist_ok=True)

    # Write cluster summaries
    for cluster_summary in summary_result.cluster_summaries:
        cluster_id = cluster_summary.cluster_id

        # Short summary
        short_path = base_dir / f"cluster_{cluster_id}.short.md"
        with open(short_path, 'w', encoding='utf-8') as f:
            f.write(f"# Cluster {cluster_id} - Short Summary\n\n")
            f.write(f"{cluster_summary.short_summary}\n")

        # Long summary
        long_path = base_dir / f"cluster_{cluster_id}.long.md"
        with open(long_path, 'w', encoding='utf-8') as f:
            f.write(f"# Cluster {cluster_id} - Detailed Summary\n\n")
            f.write(f"{cluster_summary.long_summary}\n")

    # Write project overview if available
    if summary_result.project_summary:
        project_path = base_dir / "project_overview.md"
        with open(project_path, 'w', encoding='utf-8') as f:
            f.write("# Project Overview\n\n")
            f.write(f"{summary_result.project_summary}\n")


def write_embeddings(
    path: Path,
    embeddings: np.ndarray,
):
    """
    Write embeddings to .npy file.

    Args:
        path: Path to output .npy file
        embeddings: Embeddings array
    """
    np.save(path, embeddings.astype(np.float32))


# ============================================================
# Docpack Generation
# ============================================================

def generate_docpack(
    parsed_files: List[ParsedFile],
    chunks: List[Chunk],
    embeddings: np.ndarray,
    cluster_result: ClusterResult,
    summary_result: SummaryResult,
    config: Optional[DocpackConfig] = None,
) -> DocpackResult:
    """
    Generate a complete docpack bundle.

    Args:
        parsed_files: List of parsed files
        chunks: List of chunks
        embeddings: Embeddings array
        cluster_result: Clustering result
        summary_result: Summarization result
        config: Docpack configuration (optional)

    Returns:
        DocpackResult with paths and metadata
    """
    if config is None:
        config = DocpackConfig()

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create docpack directory
    docpack_dir_name = config.docpack_name.replace(".zip", "")
    docpack_dir = output_dir / docpack_dir_name

    # Remove existing docpack directory if it exists
    if docpack_dir.exists():
        shutil.rmtree(docpack_dir)

    docpack_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("STAGE 7: DOCPACK")
    print("=" * 70)
    print(f"\nGenerating docpack: {docpack_dir_name}")

    # Generate metadata
    print("  • Generating metadata...")
    metadata = generate_metadata(config, parsed_files, chunks, embeddings, cluster_result)
    write_json_file(docpack_dir / "metadata.json", metadata)

    # Generate files.json
    print("  • Generating files.json...")
    files_json = generate_files_json(parsed_files)
    write_json_file(docpack_dir / "files.json", files_json)

    # Generate chunks.json
    print("  • Generating chunks.json...")
    chunks_json = generate_chunks_json(chunks, parsed_files, cluster_result)
    write_json_file(docpack_dir / "chunks.json", chunks_json)

    # Generate clusters.json
    print("  • Generating clusters.json...")
    clusters_json = generate_clusters_json(cluster_result)
    write_json_file(docpack_dir / "clusters.json", clusters_json)

    # Write embeddings
    print("  • Writing embeddings.npy...")
    write_embeddings(docpack_dir / "embeddings.npy", embeddings)

    # Write text files
    print("  • Writing text files...")
    write_text_files(docpack_dir / "text", parsed_files, chunks)

    # Write summaries
    print("  • Writing summaries...")
    write_summary_files(docpack_dir / "summaries", summary_result)

    # Calculate total size
    total_size = sum(f.stat().st_size for f in docpack_dir.rglob('*') if f.is_file())

    print(f"\n✓ Docpack structure created: {docpack_dir}")
    print(f"  Total size: {total_size / 1024 / 1024:.1f} MB")

    # Compress to zip if requested
    docpack_path = str(docpack_dir)
    if config.compress_to_zip:
        print(f"\n  Compressing to ZIP...")
        zip_path = output_dir / f"{docpack_dir_name}.zip"

        # Remove existing zip if it exists
        if zip_path.exists():
            zip_path.unlink()

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in docpack_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(docpack_dir.parent)
                    zipf.write(file_path, arcname)

        zip_size = zip_path.stat().st_size
        compression_ratio = (1 - zip_size / total_size) * 100 if total_size > 0 else 0

        print(f"  ✓ Compressed: {zip_path}")
        print(f"    ZIP size: {zip_size / 1024 / 1024:.1f} MB")
        print(f"    Compression: {compression_ratio:.1f}%")

        docpack_path = str(zip_path)

        # Remove uncompressed directory if requested
        if not config.keep_uncompressed:
            shutil.rmtree(docpack_dir)
            print(f"  • Removed uncompressed directory")

    print(f"\n✓ Docpack generation complete!")

    return DocpackResult(
        docpack_path=docpack_path,
        docpack_dir=str(docpack_dir),
        total_size_bytes=total_size,
        files_count=len(parsed_files),
        chunks_count=len(chunks),
        clusters_count=cluster_result.n_clusters,
        metadata=metadata,
    )
