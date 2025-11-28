# Doctown v11

Doctown is a Python-native, GPU-accelerated documentation synthesis
pipeline designed for clarity, modularity, and rapid iteration.  
This document outlines the overall architecture, the purpose of each
subsystem, and the clean responsibility boundaries that keep the project
maintainable as it grows.

The v11 design philosophy:

- **One language** (Python)
- **One toolchain** (`uv`)
- **One orchestrator**
- **GPU-first AI pipeline**
- **Strict modular boundaries**
- **No cross-layer smearing of responsibilities**

## Architecture (src/)

- src/
  - ingest/
  - parse/
  - chunk/
  - embed/
  - cluster/
  - summarize/
  - docpack/
  - main.py

---

## ingest/
### Status: Complete
Purpose: Load input repositories or ZIP files into a normalized virtual file system (VFS) without writing anything to disk.

Responsibilities:
- Accept path/URL/ZIP bytes
- Flatten ZIP archives into an in-memory file map
- Apply ignore rules (.gitignore, size limits, binary detection)
- Output: List of {path, bytes}

Non-responsibilities:
- No decoding of text
- No semantic analysis
- No chunking

---

## parse/
### Status: Complete
Purpose: Convert raw file bytes into structured, normalized UTF-8 text + semantic units.

Responsibilities:
- Detect file type heuristically
- Run appropriate parser (plain text fallback, returns just the name of exec. if binary or blob-like)
- Produce:
  - normalized_text
  - metadata (path, ext, language, size, line count)
  - semantic_units (lightweight code/document structure)

Non-responsibilities:
- No chunk merging/splitting
- No embeddings

---

## chunk/
### Status: Complete
Purpose: Convert semantic units into embedding-ready chunks under token limits.

Responsibilities:
- Merge small units
- Split large units
- Maintain semantic boundaries where possible
- Output: List of text chunks (string)

Non-responsibilities:
- No embeddings
- No model calls

---

## embed/
### Status: Complete
Purpose: GPU-accelerated embeddings via SentenceTransformers or local HF models.

Responsibilities:
- Batch embedding
- Handle CUDA/CPU fallback
- Expose clean function: embed(text_chunks) -> np.ndarray of shape (N, D)

Non-responsibilities:
- No chunking or clustering logic

---

## cluster/
### Status: Complete
Purpose: Group embeddings into coherent topic clusters.

Responsibilities:
- k-means or HDBSCAN
- cosine distance support
- return cluster assignments + cluster centroids

Non-responsibilities:
- No summarization
- No embeddings

---

## summarize/
### Status: Complete
Purpose: Generate natural-language summaries for each cluster and optionally for the whole project.

Responsibilities:
- Handle prompt templates
- Use local LLM (Qwen/Qwen3-1.7B) for summarization
- Output short and long-form summaries
- Generate project-level overview from cluster summaries

Non-responsibilities:
- No clustering, no chunking

---

## docpack/
### Status: Complete
Purpose: Assemble final documentation bundle.

Responsibilities:
- Store metadata, chunks, embeddings, clusters, and summaries
- Produce a portable docpack structure (zip, sqlite, or directory)

Non-responsibilities:
- No model inference

```txt
docpack/
  metadata.json
  files.json
  chunks.json
  embeddings.npy
  clusters.json
  summaries/
      cluster_0.md
      cluster_1.md
      project_overview.md
  text/
      file1.txt
      file2.txt
      ...
```

### metadata.json:
Global metadata for the run.

```json
{
  "docpack_version": "11",
  "generator": "doctown-v11",
  "generated_at": "2025-11-27T07:18:00Z",

  "source": {
    "type": "github",
    "owner": "xandwr",
    "repo": "localdoc",
    "branch": "main",
    "commit": "a1b2c3d4"
  },

  "stats": {
    "files": 10,
    "lines": 4539,
    "semantic_units": 405,
    "chunks": 74,
    "embedding_dim": 768,
    "clusters": 2
  },

  "pipeline": {
    "embed_model": "google/embeddinggemma-300m",
    "summarizer_model": "Qwen/Qwen3-1.7B",
    "chunk_target_tokens": 300,
    "chunk_min_tokens": 50,
    "chunk_max_tokens": 512
  }
}
```

### files.json
Describes original ingested files (VFS entries).

```json
[
  {
    "id": 0,
    "path": "src/main.rs",
    "file_type": "rust",
    "language": "rust",
    "size_bytes": 1234,
    "line_count": 87,
    "semantic_units": 24
  },
  {
    "id": 1,
    "path": "README.md",
    "file_type": "markdown",
    "language": "markdown",
    "size_bytes": 542,
    "line_count": 31,
    "semantic_units": 6
  }
]
```

### chunks.json
This describes all semantic chunks (after parsing + chunking).

```json
[
  {
    "id": 0,
    "file_id": 0,
    "cluster_id": 1,
    "start_line": 1,
    "end_line": 25,
    "token_count": 310,
    "unit_ids": [0, 1, 2, 3],
    "text_path": "text/chunk_0.txt"
  },
  {
    "id": 1,
    "file_id": 0,
    "cluster_id": 0,
    "start_line": 26,
    "end_line": 50,
    "token_count": 275,
    "unit_ids": [4, 5],
    "text_path": "text/chunk_1.txt"
  }
]
```

Chunks are written to:
text/chunk_0.txt
text/chunk_1.txt
...

No embeddings, just structure.

### embeddings.npy
A NumPy array (float32) with shape:
```py
(num_chunks, embedding_dim)
```

The embedding pipeline already produces this in Python, so saving to .npy is trivial (hopefully).
Rust, JS, Python, Julia, C++ — all can read .npy afaik.

### clusters.json
Cluster assignments + centroids metadata.

```json
{
  "algorithm": "kmeans",
  "k": 2,
  "silhouette_score": 0.569,
  "clusters": [
    {
      "id": 0,
      "chunk_ids": [1, 4, 5, 9, ...],
      "size": 57
    },
    {
      "id": 1,
      "chunk_ids": [0, 3, 7, ...],
      "size": 17
    }
  ]
}
```

Only store metadata here (assignments, sizes).
Centroids can be computed on the fly quickly if ever needed.

### summaries/
Cluster-level and project-level summaries.

```txt
summaries/
  cluster_0.short.md
  cluster_0.long.md
  cluster_1.short.md
  cluster_1.long.md
  project_overview.md
```

As long as it’s Markdown, everything (terminals, browsers, CLIs) can show it.

### text/
This directory stores:
  - the original normalized file text (file_0.txt, file_1.txt, …)
  - chunk text (chunk_0.txt, etc.)

```json
text/files/
  0.txt
  1.txt
  ...
text/chunks/
  0.txt
  1.txt
  ...
```

### Final Structure: 
```
docpack/
  metadata.json
  files.json
  chunks.json
  clusters.json
  embeddings.npy

  summaries/
    cluster_0.short.md
    cluster_0.long.md
    cluster_1.short.md
    cluster_1.long.md
    project_overview.md

  text/
    files/
      0.txt
      1.txt
      ...
    chunks/
      0.txt
      1.txt
      ...
```
Then it gets neatly zipped for output as: `project.docpack.zip`

---

## main.py
### Status: Complete (up to summarize stage)
Top-level orchestrator.

Responsibilities:
- Call subsystems in order:
  ingest → parse → chunk → embed → cluster → summarize → docpack
- Provide CLI interface
- Handle config + logging
- Print pipeline stats

