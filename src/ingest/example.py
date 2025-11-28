"""
Example usage of the ingest subsystem.
"""

from ingest import ingest_local_path, ingest_zip_bytes, ingest_github_repo

# Example 1: Ingest a local directory
print("Example 1: Local directory ingestion")
files = ingest_local_path("../../")
print(f"Loaded {len(files)} files from local path")
for f in files[:3]:
    print(f"  {f.path}: {len(f.content)} bytes")

# Example 2: Ingest a GitHub repository
print("\nExample 2: GitHub repository ingestion")
try:
    files = ingest_github_repo("octocat", "Hello-World", "master")
    print(f"Loaded {len(files)} files from GitHub")
    for f in files[:3]:
        print(f"  {f.path}: {len(f.content)} bytes")
except Exception as e:
    print(f"Failed: {e}")

# Example 3: Ingest ZIP bytes with custom filtering
print("\nExample 3: ZIP bytes with custom ignore patterns")
import requests
zip_bytes = requests.get("https://github.com/octocat/Hello-World/archive/refs/heads/master.zip").content
files = ingest_zip_bytes(
    zip_bytes,
    ignore_patterns=["*.md", "test/*"],  # Custom patterns
    max_file_size=1024 * 1024,  # 1MB limit
    filter_binary=True,
)
print(f"Loaded {len(files)} files from ZIP")

# Example 4: Access file content
print("\nExample 4: Accessing file content")
for file_entry in files:
    if file_entry.path.endswith(".py"):
        # Content is raw bytes - decode when needed
        text = file_entry.content.decode("utf-8", errors="ignore")
        print(f"Python file: {file_entry.path}")
        print(f"First 100 chars: {text[:100]}")
        break
