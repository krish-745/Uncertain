"""Golden tests: the CLI's text output for each tests/golden/*.calc must match its .expected.txt.

To update an expected file after an intentional output change, run:
    uv run uncertain tests/golden/<name>.calc > tests/golden/<name>.expected.txt
"""
import argparse
from pathlib import Path
import pytest
from uncertain.cli import run

GOLDEN_DIR = Path(__file__).parent / "golden"
CASES = sorted(GOLDEN_DIR.glob("*.calc"))

@pytest.mark.parametrize("path", CASES, ids=[p.stem for p in CASES])
def test_golden(path, capsys):
    args = argparse.Namespace(file=str(path), output="text", check_only=False, max_unroll=1000)
    run(args)
    actual = capsys.readouterr().out
    expected = path.with_suffix(".expected.txt").read_text(encoding="utf-8")
    assert actual.strip() == expected.strip()
