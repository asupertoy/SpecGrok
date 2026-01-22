
import os
import sys
from pathlib import Path

# Add src to path
# Assumes the script is located at project/SpecGrok/tests/test_md_parser.py
current_dir = Path(__file__).resolve().parent
# Go up to SpecGrok root then down to src
sys.path.append(str(current_dir.parent / "src"))

from ingestion.parsers.parser_md import MarkdownParser
from ingestion.loaders import Blob

def test_markdown_parser():
    # Construct a tricky markdown content
    md_content = """
# Introduction
This is the intro.

## Code Section
Here is some code:
```python
# This is a comment that looks like a header
def foo():
    return "bar"
```

## Math Section
Here is a formula:
$$
E = mc^2
# This should not be a header
$$

## Table Section
| Name | Age |
|------|-----|
| Alice| 30  |
| #Bob | 20  |

## Image Section
Check this out:
![Diagram](http://example.com/img.png "My Diagram")

# Conclusion
Final thoughts.
    """.strip()

    # Create a Blob
    blob = Blob(
        data=md_content.encode("utf-8"),
        source="test_doc.md"
    )

    parser = MarkdownParser(remove_images=True)
    nodes = parser.parse(blob)

    print(f"Total Nodes: {len(nodes)}")
    for i, node in enumerate(nodes):
        print(f"\n--- Node {i} ---")
        print(f"Header: {node.metadata.get('section_header')}")
        print(f"Path: {node.metadata.get('section_path')}")
        print(f"Images: {node.metadata.get('images')}")
        print(f"Content Preview:\n{node.text[:200]}")

if __name__ == "__main__":
    test_markdown_parser()
