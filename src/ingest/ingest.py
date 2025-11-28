# ingest.py
# Doctown v11 – Flattened VFS ZIP Ingestion (Python Edition)

import io
import zipfile
import requests
from dataclasses import dataclass
from pathlib import PurePath


# ============================================================
# Exceptions
# ============================================================

class IngestError(Exception):
    pass


# ============================================================
# FileEntry + VFS Representation
# ============================================================

@dataclass
class FileEntry:
    virtual_path: str
    offset: int
    length: int
    size_bytes: int
    is_utf8: bool


class VirtualFileSystem:
    """
    Flattened, sandboxed in-memory filesystem backed by one giant arena of bytes.
    """
    def __init__(self):
        self.arena = bytearray()
        self.index = {}  # path -> FileEntry

    def add_file(self, virtual_path: str, data: bytes):
        # Store metadata
        offset = len(self.arena)
        length = len(data)

        # Append to arena
        self.arena.extend(data)

        # UTF-8 detection
        try:
            data.decode("utf-8")
            is_utf8 = True
        except UnicodeDecodeError:
            is_utf8 = False

        # Record entry
        self.index[virtual_path] = FileEntry(
            virtual_path=virtual_path,
            offset=offset,
            length=length,
            size_bytes=length,
            is_utf8=is_utf8,
        )

    def get(self, virtual_path: str) -> bytes | None:
        entry = self.index.get(virtual_path)
        if not entry:
            return None
        return bytes(self.arena[entry.offset : entry.offset + entry.length])

    def list_files(self):
        return list(self.index.values())


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
# ZIP Ingestion Core
# ============================================================

def load_zip_bytes_from_url(url: str) -> bytes:
    try:
        resp = requests.get(url, timeout=20)
        if not resp.ok:
            raise IngestError(f"HTTP {resp.status_code}: failed to download {url}")
        return resp.content
    except Exception as e:
        raise IngestError(f"Failed to fetch ZIP: {e}")


def ingest_zip_bytes(zip_bytes: bytes) -> VirtualFileSystem:
    vfs = VirtualFileSystem()

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            for name in z.namelist():
                # Skip directories
                if name.endswith("/"):
                    continue

                # Strip GitHub's top-level folder
                parts = name.split("/", 1)
                stripped = parts[1] if len(parts) == 2 else name

                if not stripped.strip():
                    continue

                # Secure path normalization
                try:
                    path = sanitize_path(stripped)
                except IngestError:
                    continue  # Skip bad files silently or re-raise if you prefer

                # Read file
                data = z.read(name)
                vfs.add_file(path, data)

        return vfs
    except Exception as e:
        raise IngestError(f"Invalid ZIP: {e}")


# ============================================================
# GitHub Repo Ingestion
# ============================================================

def ingest_github_repo(owner: str, repo: str, branch: str = "main") -> VirtualFileSystem:
    url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"
    zip_bytes = load_zip_bytes_from_url(url)
    return ingest_zip_bytes(zip_bytes)


# ============================================================
# Standalone Test
# ============================================================

if __name__ == "__main__":
    print("Ingesting rust-lang/serde (main)...")

    vfs = ingest_github_repo("rust-lang", "serde", "master")

    print(f"Loaded {len(vfs.index)} files into VFS")
    print(f"Arena size: {len(vfs.arena)} bytes")

    # Example file access
    if "Cargo.toml" in vfs.index:
        cargo = vfs.get("Cargo.toml").decode("utf-8")
        print("\nCargo.toml preview:\n", cargo[:300])
