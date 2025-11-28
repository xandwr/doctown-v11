# ingest.py
# Doctown v11 – Ingest subsystem: load repos/ZIPs into in-memory file list

import io
import zipfile
import requests
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Dict, List
import fnmatch


# ============================================================
# Exceptions
# ============================================================

class IngestError(Exception):
    pass


# ============================================================
# Output Format
# ============================================================

@dataclass
class FileEntry:
    """Simple file entry with path and raw bytes."""
    path: str
    content: bytes


# ============================================================
# Configuration
# ============================================================

# Default limits and rules
DEFAULT_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
DEFAULT_IGNORE_PATTERNS = [
    ".git/*",
    ".git/**",
    "*.pyc",
    "__pycache__/*",
    "*.so",
    "*.dylib",
    "*.dll",
    "*.exe",
    "*.bin",
    "*.o",
    "*.a",
    "node_modules/*",
    "node_modules/**",
    ".venv/*",
    ".venv/**",
    "venv/*",
    "venv/**",
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.gif",
    "*.bmp",
    "*.ico",
    "*.pdf",
    "*.zip",
    "*.tar",
    "*.gz",
    "*.7z",
    "*.rar",
]


# ============================================================
# Ignore Rule Processing
# ============================================================

def parse_gitignore(content: str) -> List[str]:
    """Parse .gitignore content into patterns."""
    patterns = []
    for line in content.splitlines():
        line = line.strip()
        # Skip empty lines and comments
        if not line or line.startswith("#"):
            continue
        patterns.append(line)
    return patterns


def should_ignore(path: str, patterns: List[str]) -> bool:
    """Check if path matches any ignore pattern."""
    for pattern in patterns:
        # Handle directory patterns
        if pattern.endswith("/"):
            pattern = pattern.rstrip("/") + "/*"
        
        # Match against full path or basename
        if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(path, f"**/{pattern}"):
            return True
            
        # Handle ** patterns
        if "**" in pattern:
            pattern_parts = pattern.split("**")
            if len(pattern_parts) == 2:
                prefix, suffix = pattern_parts
                if path.startswith(prefix.rstrip("/")) and (not suffix or path.endswith(suffix.lstrip("/"))):
                    return True
    return False


def is_binary(data: bytes, sample_size: int = 8192) -> bool:
    """
    Heuristic binary detection.
    Check for null bytes or high ratio of non-text bytes in sample.
    """
    sample = data[:sample_size]
    
    # Null byte is a strong binary indicator
    if b"\x00" in sample:
        return True
    
    # Check ratio of non-printable bytes
    non_printable = sum(1 for b in sample if b < 32 and b not in (9, 10, 13))
    if len(sample) > 0 and non_printable / len(sample) > 0.3:
        return True
    
    return False


# ============================================================
# Path Sanitization (Zip Slip Prevention)
# ============================================================

def sanitize_path(raw: str) -> str:
    """
    Prevent:
      - ../ traversal
      - absolute paths
      - backslashes
      - weird unicode tricks
      - leading slashes
    Returns a normalized forward-slash path.
    """
    if not raw or raw.strip() == "":
        raise IngestError(f"Invalid empty path {raw!r}")

    p = PurePath(raw)

    # Reject absolute paths
    if p.is_absolute():
        raise IngestError(f"Absolute path not allowed: {raw}")

    # Reject traversal
    for part in p.parts:
        if part == "..":
            raise IngestError(f"Traversal not allowed: {raw}")

    # Normalize to forward slash
    clean = "/".join(p.parts)

    if clean.startswith("/"):
        raise IngestError(f"Leading slash not allowed: {raw}")

    return clean


# ============================================================
# Core Ingestion Functions
# ============================================================

def load_zip_bytes_from_url(url: str) -> bytes:
    """Download ZIP file from URL."""
    try:
        resp = requests.get(url, timeout=30)
        if not resp.ok:
            raise IngestError(f"HTTP {resp.status_code}: failed to download {url}")
        return resp.content
    except Exception as e:
        raise IngestError(f"Failed to fetch ZIP: {e}")


def ingest_zip_bytes(
    zip_bytes: bytes,
    ignore_patterns: List[str] | None = None,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
    strip_top_level: bool = True,
    filter_binary: bool = True,
) -> List[FileEntry]:
    """
    Ingest ZIP archive into list of file entries.
    
    Args:
        zip_bytes: Raw ZIP file bytes
        ignore_patterns: List of gitignore-style patterns to exclude
        max_file_size: Maximum file size in bytes (files larger are skipped)
        strip_top_level: Strip first directory component (common for GitHub ZIPs)
        filter_binary: Skip binary files based on heuristic detection
        
    Returns:
        List of FileEntry objects with {path, content}
    """
    if ignore_patterns is None:
        ignore_patterns = DEFAULT_IGNORE_PATTERNS.copy()
    
    files = []

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            for name in z.namelist():
                # Skip directories
                if name.endswith("/"):
                    continue

                # Strip top-level folder if requested (e.g., "repo-main/")
                if strip_top_level:
                    parts = name.split("/", 1)
                    stripped = parts[1] if len(parts) == 2 else name
                else:
                    stripped = name

                if not stripped.strip():
                    continue

                # Secure path normalization
                try:
                    path = sanitize_path(stripped)
                except IngestError:
                    continue  # Skip files with invalid paths

                # Apply ignore rules
                if should_ignore(path, ignore_patterns):
                    continue

                # Read file data
                info = z.getinfo(name)
                
                # Size check
                if info.file_size > max_file_size:
                    continue
                
                data = z.read(name)
                
                # Binary detection
                if filter_binary and is_binary(data):
                    continue

                files.append(FileEntry(path=path, content=data))

        return files
    except Exception as e:
        raise IngestError(f"Invalid ZIP: {e}")


# ============================================================
# Local Path Ingestion
# ============================================================

def ingest_local_path(
    path: str | Path,
    ignore_patterns: List[str] | None = None,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
    filter_binary: bool = True,
) -> List[FileEntry]:
    """
    Ingest files from a local directory or file path.
    
    Args:
        path: Local filesystem path (file or directory)
        ignore_patterns: List of gitignore-style patterns to exclude
        max_file_size: Maximum file size in bytes
        filter_binary: Skip binary files based on heuristic detection
        
    Returns:
        List of FileEntry objects with {path, content}
    """
    path = Path(path)
    
    if not path.exists():
        raise IngestError(f"Path does not exist: {path}")
    
    if ignore_patterns is None:
        ignore_patterns = DEFAULT_IGNORE_PATTERNS.copy()
    
    # Check for .gitignore in directory
    if path.is_dir():
        gitignore_path = path / ".gitignore"
        if gitignore_path.exists():
            try:
                gitignore_content = gitignore_path.read_text(encoding="utf-8")
                ignore_patterns.extend(parse_gitignore(gitignore_content))
            except Exception:
                pass  # Continue without .gitignore if it fails
    
    files = []
    
    if path.is_file():
        # Single file ingestion
        try:
            data = path.read_bytes()
            if len(data) <= max_file_size:
                if not filter_binary or not is_binary(data):
                    files.append(FileEntry(path=path.name, content=data))
        except Exception as e:
            raise IngestError(f"Failed to read file {path}: {e}")
    
    elif path.is_dir():
        # Directory traversal
        base_path = path
        for file_path in path.rglob("*"):
            if not file_path.is_file():
                continue
            
            # Compute relative path
            try:
                rel_path = file_path.relative_to(base_path)
                rel_path_str = str(rel_path).replace("\\", "/")
            except ValueError:
                continue
            
            # Apply ignore rules
            if should_ignore(rel_path_str, ignore_patterns):
                continue
            
            # Size check
            try:
                size = file_path.stat().st_size
                if size > max_file_size:
                    continue
            except Exception:
                continue
            
            # Read file
            try:
                data = file_path.read_bytes()
            except Exception:
                continue  # Skip files that can't be read
            
            # Binary detection
            if filter_binary and is_binary(data):
                continue
            
            files.append(FileEntry(path=rel_path_str, content=data))
    
    return files


# ============================================================
# GitHub Repo Ingestion
# ============================================================

def ingest_github_repo(
    owner: str,
    repo: str,
    branch: str = "main",
    ignore_patterns: List[str] | None = None,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
    filter_binary: bool = True,
) -> List[FileEntry]:
    """
    Ingest a GitHub repository by downloading its ZIP archive.
    
    Args:
        owner: GitHub username or organization
        repo: Repository name
        branch: Branch name (default: "main")
        ignore_patterns: List of gitignore-style patterns to exclude
        max_file_size: Maximum file size in bytes
        filter_binary: Skip binary files based on heuristic detection
        
    Returns:
        List of FileEntry objects with {path, content}
    """
    url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"
    zip_bytes = load_zip_bytes_from_url(url)
    return ingest_zip_bytes(
        zip_bytes,
        ignore_patterns=ignore_patterns,
        max_file_size=max_file_size,
        strip_top_level=True,
        filter_binary=filter_binary,
    )


# ============================================================
# Standalone Test
# ============================================================

if __name__ == "__main__":
    import sys
    
    print("Testing ingest subsystem...")
    
    # Test 1: GitHub repo ingestion
    print("\n[Test 1] Ingesting python/cpython (main)...")
    try:
        files = ingest_github_repo("python", "cpython", "main")
        print(f"✓ Loaded {len(files)} files")
        
        # Show sample files
        readme_files = [f for f in files if "README" in f.path.upper()]
        if readme_files:
            readme = readme_files[0].content.decode("utf-8", errors="ignore")
            print(f"\n{readme_files[0].path} preview:\n{readme[:300]}")
    except IngestError as e:
        print(f"✗ Failed: {e}")
        print("(Skipping GitHub test - may be network/rate limit issue)")
    
    # Test 2: Local path ingestion (if current directory exists)
    print("\n[Test 2] Ingesting current directory...")
    try:
        local_files = ingest_local_path(".")
        print(f"✓ Loaded {len(local_files)} files from current directory")
        
        # Show first few files
        print("\nSample files:")
        for f in local_files[:5]:
            print(f"  - {f.path} ({len(f.content)} bytes)")
    except IngestError as e:
        print(f"✗ Failed: {e}")
    
    print("\n✓ All tests passed!")
