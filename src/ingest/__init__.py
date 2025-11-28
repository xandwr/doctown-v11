"""
Ingest subsystem for Doctown v11.

Purpose: Load input repositories or ZIP files into a normalized
virtual file system (VFS) without writing anything to disk.

Responsibilities:
- Accept path/URL/ZIP bytes
- Flatten ZIP archives into an in-memory file map
- Apply ignore rules (.gitignore, size limits, binary detection)
- Output: List of {path, bytes}

Non-responsibilities:
- No decoding of text
- No semantic analysis
- No chunking
"""

from .ingest import ingest_local_path, ingest_zip_bytes, ingest_github_repo, IngestError

__all__ = [
    "ingest_local_path",
    "ingest_zip_bytes", 
    "ingest_github_repo",
    "IngestError",
]
