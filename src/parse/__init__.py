"""
Parse subsystem for Doctown v11.

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
"""

from .parse import (
    parse_file,
    parse_files,
    ParsedFile,
    SemanticUnit,
    FileType,
    UnitType,
    ParseError,
)

__all__ = [
    "parse_file",
    "parse_files",
    "ParsedFile",
    "SemanticUnit",
    "FileType",
    "UnitType",
    "ParseError",
]
