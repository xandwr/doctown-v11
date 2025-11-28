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
Purpose: Convert raw file bytes into structured, normalized UTF-8 text + semantic units.

Responsibilities:
- Detect file type heuristically
- Run appropriate parser (plain text fallback)
- Produce:
  - normalized_text
  - metadata (path, ext, language, size, line count)
  - semantic_units (lightweight code/document structure)

Non-responsibilities:
- No chunk merging/splitting
- No embeddings

---

## chunk/
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
Purpose: GPU-accelerated embeddings via SentenceTransformers or local HF models.

Responsibilities:
- Batch embedding
- Handle CUDA/CPU fallback
- Expose clean function: embed(text_chunks) -> np.ndarray of shape (N, D)

Non-responsibilities:
- No chunking or clustering logic

---

## cluster/
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
Purpose: Generate natural-language summaries for each cluster and optionally for the whole project.

Responsibilities:
- Handle prompt templates
- Use local LLM (e.g., Qwen, DeepSeek) for summarization
- Output short and long-form summaries

Non-responsibilities:
- No clustering, no chunking

---

## docpack/
Purpose: Assemble final documentation bundle.

Responsibilities:
- Store metadata, chunks, embeddings, clusters, and summaries
- Produce a portable docpack structure (zip, sqlite, or directory)

Non-responsibilities:
- No model inference

---

## main.py
Top-level orchestrator.

Responsibilities:
- Call subsystems in order:
  ingest → parse → chunk → embed → cluster → summarize → docpack
- Provide CLI interface
- Handle config + logging
- Print pipeline stats

