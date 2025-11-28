"""
Example usage of the parse subsystem with ingest integration.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest import ingest_local_path
from parse import parse_files, FileType, UnitType


def main():
    print("Parse subsystem integration example\n")
    
    # Step 1: Ingest files from current directory
    print("[1] Ingesting files from src/parse/...")
    file_entries = ingest_local_path(".", max_file_size=100_000)
    print(f"✓ Ingested {len(file_entries)} files")
    
    # Step 2: Parse all files
    print("\n[2] Parsing files...")
    parsed_files = parse_files(file_entries)
    print(f"✓ Parsed {len(parsed_files)} files")
    
    # Step 3: Show statistics
    print("\n[3] Statistics:")
    type_counts = {}
    for pf in parsed_files:
        type_counts[pf.file_type] = type_counts.get(pf.file_type, 0) + 1
    
    print("File types:")
    for file_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {file_type.value}: {count}")
    
    total_lines = sum(pf.line_count for pf in parsed_files)
    total_units = sum(len(pf.semantic_units) for pf in parsed_files)
    print(f"\nTotal lines: {total_lines}")
    print(f"Total semantic units: {total_units}")
    
    # Step 4: Show semantic units from Python files
    print("\n[4] Semantic units from Python files:")
    for pf in parsed_files:
        if pf.file_type == FileType.PYTHON:
            print(f"\n{pf.path}:")
            unit_counts = {}
            for unit in pf.semantic_units:
                unit_counts[unit.type] = unit_counts.get(unit.type, 0) + 1
            
            for unit_type, count in unit_counts.items():
                print(f"  {unit_type.value}: {count}")
            
            # Show first few units
            print(f"  Sample units:")
            for unit in pf.semantic_units[:5]:
                print(f"    - {unit.type.value} '{unit.name}' at line {unit.line_start}")
    
    # Step 5: Show output format
    print("\n[5] Sample ParsedFile structure:")
    if parsed_files:
        pf = parsed_files[0]
        print(f"  path: {pf.path}")
        print(f"  file_type: {pf.file_type.value}")
        print(f"  language: {pf.language}")
        print(f"  size_bytes: {pf.size_bytes}")
        print(f"  line_count: {pf.line_count}")
        print(f"  semantic_units: {len(pf.semantic_units)} units")
        print(f"  normalized_text: {len(pf.normalized_text)} chars")
        
        if pf.semantic_units:
            unit = pf.semantic_units[0]
            print(f"\n  Sample unit:")
            print(f"    type: {unit.type.value}")
            print(f"    name: {unit.name}")
            print(f"    lines: {unit.line_start}-{unit.line_end}")
            print(f"    content preview: {unit.content[:100]!r}...")
    
    print("\n✓ Integration example complete!")


if __name__ == "__main__":
    main()
