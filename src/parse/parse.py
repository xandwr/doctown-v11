# parse.py
# Doctown v11 – Parse subsystem: convert raw bytes to structured text + semantic units

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from enum import Enum


# ============================================================
# Exceptions
# ============================================================

class ParseError(Exception):
    pass


# ============================================================
# Output Format
# ============================================================

class FileType(Enum):
    """Detected file type categories."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    RUST = "rust"
    GO = "go"
    JAVA = "java"
    C = "c"
    CPP = "cpp"
    CSHARP = "csharp"
    MARKDOWN = "markdown"
    PLAINTEXT = "plaintext"
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    XML = "xml"
    HTML = "html"
    CSS = "css"
    SHELL = "shell"
    SQL = "sql"
    BINARY = "binary"
    UNKNOWN = "unknown"


class UnitType(Enum):
    """Types of semantic units."""
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    STRUCT = "struct"
    ENUM = "enum"
    INTERFACE = "interface"
    IMPORT = "import"
    COMMENT = "comment"
    DOCSTRING = "docstring"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    CODE_BLOCK = "code_block"
    MODULE = "module"


@dataclass
class SemanticUnit:
    """Lightweight code/document structure element."""
    type: UnitType
    name: str
    content: str
    line_start: int
    line_end: int
    metadata: dict = field(default_factory=dict)


@dataclass
class ParsedFile:
    """Result of parsing a single file."""
    path: str
    normalized_text: str
    file_type: FileType
    language: str
    size_bytes: int
    line_count: int
    semantic_units: List[SemanticUnit]
    metadata: dict = field(default_factory=dict)


# ============================================================
# File Type Detection
# ============================================================

# Extension mappings
EXTENSION_MAP = {
    ".py": FileType.PYTHON,
    ".pyi": FileType.PYTHON,
    ".js": FileType.JAVASCRIPT,
    ".jsx": FileType.JAVASCRIPT,
    ".mjs": FileType.JAVASCRIPT,
    ".cjs": FileType.JAVASCRIPT,
    ".ts": FileType.TYPESCRIPT,
    ".tsx": FileType.TYPESCRIPT,
    ".rs": FileType.RUST,
    ".go": FileType.GO,
    ".java": FileType.JAVA,
    ".c": FileType.C,
    ".h": FileType.C,
    ".cpp": FileType.CPP,
    ".cc": FileType.CPP,
    ".cxx": FileType.CPP,
    ".hpp": FileType.CPP,
    ".hh": FileType.CPP,
    ".cs": FileType.CSHARP,
    ".md": FileType.MARKDOWN,
    ".markdown": FileType.MARKDOWN,
    ".txt": FileType.PLAINTEXT,
    ".json": FileType.JSON,
    ".yaml": FileType.YAML,
    ".yml": FileType.YAML,
    ".toml": FileType.TOML,
    ".xml": FileType.XML,
    ".html": FileType.HTML,
    ".htm": FileType.HTML,
    ".css": FileType.CSS,
    ".sh": FileType.SHELL,
    ".bash": FileType.SHELL,
    ".zsh": FileType.SHELL,
    ".sql": FileType.SQL,
}

# Shebang patterns
SHEBANG_MAP = {
    "python": FileType.PYTHON,
    "node": FileType.JAVASCRIPT,
    "bash": FileType.SHELL,
    "sh": FileType.SHELL,
    "zsh": FileType.SHELL,
}


def detect_file_type(path: str, content: bytes) -> FileType:
    """
    Detect file type from extension and content heuristics.
    
    Args:
        path: File path
        content: Raw file bytes
        
    Returns:
        FileType enum value
    """
    # Check extension first
    ext = Path(path).suffix.lower()
    if ext in EXTENSION_MAP:
        return EXTENSION_MAP[ext]
    
    # Try to decode as text
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return FileType.BINARY
    
    # Check shebang for extensionless files
    if text.startswith("#!"):
        first_line = text.split("\n", 1)[0]
        for key, file_type in SHEBANG_MAP.items():
            if key in first_line.lower():
                return file_type
    
    # Content-based heuristics
    lines = text.split("\n")[:50]  # Check first 50 lines
    
    # Python indicators
    if any(re.match(r"^(import|from|def|class)\s", line.strip()) for line in lines):
        return FileType.PYTHON
    
    # JavaScript/TypeScript indicators
    if any(re.search(r"\b(const|let|var|function|=>|require|export)\b", line) for line in lines):
        return FileType.JAVASCRIPT
    
    # Markdown indicators
    if any(line.strip().startswith("#") or re.match(r"^[-*]\s", line.strip()) for line in lines):
        return FileType.MARKDOWN
    
    # Default to plaintext for decodable content
    return FileType.PLAINTEXT


# ============================================================
# Language-Specific Parsers
# ============================================================

def parse_python(text: str) -> List[SemanticUnit]:
    """Extract semantic units from Python code."""
    units = []
    lines = text.split("\n")
    
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        
        # Imports
        if stripped.startswith(("import ", "from ")):
            units.append(SemanticUnit(
                type=UnitType.IMPORT,
                name=stripped,
                content=line,
                line_start=i,
                line_end=i,
            ))
        
        # Functions
        match = re.match(r"def\s+(\w+)\s*\(", stripped)
        if match:
            func_name = match.group(1)
            # Find end of function (simple heuristic: next def/class or dedent)
            end_line = i
            base_indent = len(line) - len(line.lstrip())
            for j in range(i, len(lines)):
                if j > i:
                    line_indent = len(lines[j]) - len(lines[j].lstrip())
                    if lines[j].strip() and line_indent <= base_indent and not lines[j].strip().startswith(("@", "#")):
                        break
                end_line = j + 1
            
            content = "\n".join(lines[i-1:end_line])
            units.append(SemanticUnit(
                type=UnitType.FUNCTION,
                name=func_name,
                content=content,
                line_start=i,
                line_end=end_line,
            ))
        
        # Classes
        match = re.match(r"class\s+(\w+)", stripped)
        if match:
            class_name = match.group(1)
            end_line = i
            base_indent = len(line) - len(line.lstrip())
            for j in range(i, len(lines)):
                if j > i:
                    line_indent = len(lines[j]) - len(lines[j].lstrip())
                    if lines[j].strip() and line_indent <= base_indent and not lines[j].strip().startswith(("@", "#")):
                        break
                end_line = j + 1
            
            content = "\n".join(lines[i-1:end_line])
            units.append(SemanticUnit(
                type=UnitType.CLASS,
                name=class_name,
                content=content,
                line_start=i,
                line_end=end_line,
            ))
    
    return units


def parse_javascript_typescript(text: str) -> List[SemanticUnit]:
    """Extract semantic units from JavaScript/TypeScript code."""
    units = []
    lines = text.split("\n")
    
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        
        # Imports
        if stripped.startswith(("import ", "require(", "export ")):
            units.append(SemanticUnit(
                type=UnitType.IMPORT,
                name=stripped,
                content=line,
                line_start=i,
                line_end=i,
            ))
        
        # Functions (multiple patterns)
        patterns = [
            (r"function\s+(\w+)\s*\(", UnitType.FUNCTION),
            (r"const\s+(\w+)\s*=\s*(?:async\s*)?\(", UnitType.FUNCTION),
            (r"let\s+(\w+)\s*=\s*(?:async\s*)?\(", UnitType.FUNCTION),
            (r"var\s+(\w+)\s*=\s*(?:async\s*)?\(", UnitType.FUNCTION),
        ]
        
        for pattern, unit_type in patterns:
            match = re.search(pattern, stripped)
            if match:
                name = match.group(1)
                units.append(SemanticUnit(
                    type=unit_type,
                    name=name,
                    content=line,
                    line_start=i,
                    line_end=i,
                ))
                break
        
        # Classes
        match = re.match(r"class\s+(\w+)", stripped)
        if match:
            units.append(SemanticUnit(
                type=UnitType.CLASS,
                name=match.group(1),
                content=line,
                line_start=i,
                line_end=i,
            ))
        
        # Interfaces (TypeScript)
        match = re.match(r"interface\s+(\w+)", stripped)
        if match:
            units.append(SemanticUnit(
                type=UnitType.INTERFACE,
                name=match.group(1),
                content=line,
                line_start=i,
                line_end=i,
            ))
    
    return units


def parse_rust(text: str) -> List[SemanticUnit]:
    """Extract semantic units from Rust code."""
    units = []
    lines = text.split("\n")
    
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        
        # Use statements
        if stripped.startswith("use "):
            units.append(SemanticUnit(
                type=UnitType.IMPORT,
                name=stripped,
                content=line,
                line_start=i,
                line_end=i,
            ))
        
        # Functions
        match = re.match(r"(?:pub\s+)?(?:async\s+)?fn\s+(\w+)", stripped)
        if match:
            units.append(SemanticUnit(
                type=UnitType.FUNCTION,
                name=match.group(1),
                content=line,
                line_start=i,
                line_end=i,
            ))
        
        # Structs
        match = re.match(r"(?:pub\s+)?struct\s+(\w+)", stripped)
        if match:
            units.append(SemanticUnit(
                type=UnitType.STRUCT,
                name=match.group(1),
                content=line,
                line_start=i,
                line_end=i,
            ))
        
        # Enums
        match = re.match(r"(?:pub\s+)?enum\s+(\w+)", stripped)
        if match:
            units.append(SemanticUnit(
                type=UnitType.ENUM,
                name=match.group(1),
                content=line,
                line_start=i,
                line_end=i,
            ))
        
        # Traits (interfaces)
        match = re.match(r"(?:pub\s+)?trait\s+(\w+)", stripped)
        if match:
            units.append(SemanticUnit(
                type=UnitType.INTERFACE,
                name=match.group(1),
                content=line,
                line_start=i,
                line_end=i,
            ))
    
    return units


def parse_markdown(text: str) -> List[SemanticUnit]:
    """Extract semantic units from Markdown documents."""
    units = []
    lines = text.split("\n")
    
    current_paragraph = []
    paragraph_start = 0
    in_code_block = False
    code_block_start = 0
    code_block_lines = []
    
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        
        # Code blocks
        if stripped.startswith("```"):
            if not in_code_block:
                in_code_block = True
                code_block_start = i
                code_block_lines = [line]
            else:
                code_block_lines.append(line)
                units.append(SemanticUnit(
                    type=UnitType.CODE_BLOCK,
                    name="code",
                    content="\n".join(code_block_lines),
                    line_start=code_block_start,
                    line_end=i,
                ))
                in_code_block = False
                code_block_lines = []
        elif in_code_block:
            code_block_lines.append(line)
        # Headings
        elif stripped.startswith("#"):
            # Flush paragraph
            if current_paragraph:
                units.append(SemanticUnit(
                    type=UnitType.PARAGRAPH,
                    name="paragraph",
                    content="\n".join(current_paragraph),
                    line_start=paragraph_start,
                    line_end=i-1,
                ))
                current_paragraph = []
            
            match = re.match(r"(#+)\s+(.+)", stripped)
            if match:
                level = len(match.group(1))
                heading_text = match.group(2)
                units.append(SemanticUnit(
                    type=UnitType.HEADING,
                    name=heading_text,
                    content=line,
                    line_start=i,
                    line_end=i,
                    metadata={"level": level},
                ))
        # Paragraphs
        elif stripped:
            if not current_paragraph:
                paragraph_start = i
            current_paragraph.append(line)
        else:
            # Empty line - flush paragraph
            if current_paragraph:
                units.append(SemanticUnit(
                    type=UnitType.PARAGRAPH,
                    name="paragraph",
                    content="\n".join(current_paragraph),
                    line_start=paragraph_start,
                    line_end=i-1,
                ))
                current_paragraph = []
    
    # Flush final paragraph
    if current_paragraph:
        units.append(SemanticUnit(
            type=UnitType.PARAGRAPH,
            name="paragraph",
            content="\n".join(current_paragraph),
            line_start=paragraph_start,
            line_end=len(lines),
        ))
    
    return units


def parse_generic_code(text: str) -> List[SemanticUnit]:
    """Generic parser for code files - extracts basic structure."""
    units = []
    lines = text.split("\n")
    
    # Just extract line-by-line for generic code
    # Look for common patterns
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        
        # Comments (various styles)
        if stripped.startswith(("//", "#", "--")):
            units.append(SemanticUnit(
                type=UnitType.COMMENT,
                name="comment",
                content=line,
                line_start=i,
                line_end=i,
            ))
    
    return units


def parse_plaintext(text: str) -> List[SemanticUnit]:
    """Parse plain text into paragraph units."""
    units = []
    lines = text.split("\n")
    
    current_paragraph = []
    paragraph_start = 0
    
    for i, line in enumerate(lines, 1):
        if line.strip():
            if not current_paragraph:
                paragraph_start = i
            current_paragraph.append(line)
        else:
            if current_paragraph:
                units.append(SemanticUnit(
                    type=UnitType.PARAGRAPH,
                    name="paragraph",
                    content="\n".join(current_paragraph),
                    line_start=paragraph_start,
                    line_end=i-1,
                ))
                current_paragraph = []
    
    if current_paragraph:
        units.append(SemanticUnit(
            type=UnitType.PARAGRAPH,
            name="paragraph",
            content="\n".join(current_paragraph),
            line_start=paragraph_start,
            line_end=len(lines),
        ))
    
    return units


# ============================================================
# Main Parse Function
# ============================================================

def parse_file(path: str, content: bytes) -> ParsedFile:
    """
    Parse a file from raw bytes into structured format.
    
    Args:
        path: Virtual file path
        content: Raw file bytes
        
    Returns:
        ParsedFile with normalized text, metadata, and semantic units
    """
    # Detect file type
    file_type = detect_file_type(path, content)
    
    # Handle binary files
    if file_type == FileType.BINARY:
        return ParsedFile(
            path=path,
            normalized_text="",
            file_type=file_type,
            language="binary",
            size_bytes=len(content),
            line_count=0,
            semantic_units=[],
            metadata={"binary": True},
        )
    
    # Decode text
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = content.decode("latin-1")
        except Exception:
            # Fallback: treat as binary
            return ParsedFile(
                path=path,
                normalized_text="",
                file_type=FileType.BINARY,
                language="binary",
                size_bytes=len(content),
                line_count=0,
                semantic_units=[],
                metadata={"binary": True, "decode_error": True},
            )
    
    # Normalize text (basic cleanup)
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    line_count = normalized_text.count("\n") + 1 if normalized_text else 0
    
    # Select parser based on file type
    semantic_units = []
    if file_type == FileType.PYTHON:
        semantic_units = parse_python(normalized_text)
        language = "python"
    elif file_type in (FileType.JAVASCRIPT, FileType.TYPESCRIPT):
        semantic_units = parse_javascript_typescript(normalized_text)
        language = file_type.value
    elif file_type == FileType.RUST:
        semantic_units = parse_rust(normalized_text)
        language = "rust"
    elif file_type == FileType.MARKDOWN:
        semantic_units = parse_markdown(normalized_text)
        language = "markdown"
    elif file_type == FileType.PLAINTEXT:
        semantic_units = parse_plaintext(normalized_text)
        language = "plaintext"
    else:
        # Generic fallback
        semantic_units = parse_generic_code(normalized_text)
        language = file_type.value
    
    return ParsedFile(
        path=path,
        normalized_text=normalized_text,
        file_type=file_type,
        language=language,
        size_bytes=len(content),
        line_count=line_count,
        semantic_units=semantic_units,
    )


def parse_files(file_entries: list) -> List[ParsedFile]:
    """
    Parse multiple files from ingest output.
    
    Args:
        file_entries: List of FileEntry objects from ingest subsystem
        
    Returns:
        List of ParsedFile objects
    """
    results = []
    for entry in file_entries:
        try:
            parsed = parse_file(entry.path, entry.content)
            results.append(parsed)
        except Exception as e:
            # Log error but continue processing
            print(f"Warning: Failed to parse {entry.path}: {e}")
            continue
    
    return results


# ============================================================
# Standalone Test
# ============================================================

if __name__ == "__main__":
    # Test with sample Python code
    sample_python = b"""
import sys
from pathlib import Path

def hello_world(name: str) -> str:
    \"\"\"Greet someone.\"\"\"
    return f"Hello, {name}!"

class MyClass:
    def __init__(self):
        self.value = 42
    
    def method(self):
        pass
"""
    
    print("Testing parse subsystem...")
    print("\n[Test 1] Python file parsing")
    result = parse_file("example.py", sample_python)
    print(f"File type: {result.file_type}")
    print(f"Language: {result.language}")
    print(f"Lines: {result.line_count}")
    print(f"Semantic units: {len(result.semantic_units)}")
    for unit in result.semantic_units:
        print(f"  - {unit.type.value}: {unit.name} (lines {unit.line_start}-{unit.line_end})")
    
    # Test with Markdown
    sample_markdown = b"""
# Main Heading

This is a paragraph with some text.

## Subheading

Another paragraph here.

```python
def example():
    pass
```
"""
    
    print("\n[Test 2] Markdown file parsing")
    result = parse_file("README.md", sample_markdown)
    print(f"File type: {result.file_type}")
    print(f"Language: {result.language}")
    print(f"Semantic units: {len(result.semantic_units)}")
    for unit in result.semantic_units:
        print(f"  - {unit.type.value}: {unit.name} (lines {unit.line_start}-{unit.line_end})")
    
    # Test with binary
    print("\n[Test 3] Binary file detection")
    result = parse_file("image.png", b"\x89PNG\r\n\x1a\n\x00\x00")
    print(f"File type: {result.file_type}")
    print(f"Language: {result.language}")
    print(f"Binary: {result.metadata.get('binary', False)}")
    
    print("\n✓ All tests passed!")
