from __future__ import annotations

import argparse
from pathlib import Path

from longdoc_transformer_classifier.visualization import plot_model_comparison


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot metrics from model_comparison.json.")
    parser.add_argument("--comparison-path", type=Path, default=Path("reports/model_comparison.json"))
    parser.add_argument("--figures-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    macro_f1_path, accuracy_path = plot_model_comparison(
        comparison_path=args.comparison_path,
        figures_dir=args.figures_dir,
    )
    print(f"Saved {macro_f1_path}")
    print(f"Saved {accuracy_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
