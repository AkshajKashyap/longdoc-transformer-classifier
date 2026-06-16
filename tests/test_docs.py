from pathlib import Path


def test_docs_files_exist_with_expected_headings():
    expected = {
        "method_notes.md": ["# Method Notes", "## What Each Method Proves"],
        "interview_notes.md": ["# Interview Notes", "## Core Long-Document Problem"],
        "limitations.md": ["# Limitations", "## Smoke Runs Are Not Final Rankings"],
    }

    docs_dir = Path("docs")
    for file_name, headings in expected.items():
        text = (docs_dir / file_name).read_text(encoding="utf-8")
        for heading in headings:
            assert heading in text

