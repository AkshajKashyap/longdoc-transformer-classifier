from __future__ import annotations

import argparse
from pathlib import Path

from longdoc_transformer_classifier.portfolio_reports import write_portfolio_reports


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write final portfolio reports.")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = write_portfolio_reports(args.reports_dir)
    for path in paths:
        print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
