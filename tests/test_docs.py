from pathlib import Path


def test_docs_files_exist_with_expected_headings():
    expected = {
        "method_notes.md": ["# Method Notes", "## What Each Method Proves"],
        "interview_notes.md": ["# Interview Notes", "## Core Long-Document Problem"],
        "limitations.md": ["# Limitations", "## Smoke Runs Are Not Final Rankings"],
        "reproducibility.md": ["# Reproducibility", "## Environment Setup"],
    }

    docs_dir = Path("docs")
    for file_name, headings in expected.items():
        text = (docs_dir / file_name).read_text(encoding="utf-8")
        for heading in headings:
            assert heading in text


def test_readme_referenced_docs_paths_exist():
    readme = Path("README.md").read_text(encoding="utf-8")
    for path in [
        "docs/method_notes.md",
        "docs/interview_notes.md",
        "docs/limitations.md",
        "docs/reproducibility.md",
    ]:
        assert path in readme
        assert Path(path).exists()


def test_ci_file_exists_and_runs_lint_and_tests():
    ci_path = Path(".github/workflows/ci.yml")
    text = ci_path.read_text(encoding="utf-8")

    assert "ruff check ." in text
    assert "pytest -q" in text
