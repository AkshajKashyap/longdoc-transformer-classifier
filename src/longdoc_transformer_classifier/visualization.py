from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "longdoc_transformer_classifier_mpl"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def plot_model_comparison(
    comparison_path: Path = Path("reports/model_comparison.json"),
    figures_dir: Path | None = None,
) -> tuple[Path, Path]:
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    base_dir = comparison_path.parent
    output_dir = figures_dir or base_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    macro_f1_path = output_dir / "model_macro_f1_by_method.png"
    accuracy_path = output_dir / "model_accuracy_by_method.png"
    rows = comparison.get("rows", [])
    _plot_metric(
        rows=rows,
        metric_key="macro_f1",
        title="Macro-F1 By Method",
        ylabel="Macro-F1",
        output_path=macro_f1_path,
    )
    _plot_metric(
        rows=rows,
        metric_key="accuracy",
        title="Accuracy By Method",
        ylabel="Accuracy",
        output_path=accuracy_path,
    )
    return macro_f1_path, accuracy_path


def _plot_metric(
    rows: list[dict[str, Any]],
    metric_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    metric_rows = [row for row in rows if row.get(metric_key) is not None]
    labels = [_row_label(row) for row in metric_rows]
    values = [float(row[metric_key]) for row in metric_rows]

    width = max(8.0, 0.8 * max(len(labels), 1))
    _, axis = plt.subplots(figsize=(width, 4.8))
    if values:
        axis.bar(labels, values)
        axis.set_ylim(0.0, min(1.0, max(values) + 0.1))
    else:
        axis.text(
            0.5,
            0.5,
            "No metrics available",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_xticks([])
        axis.set_ylim(0.0, 1.0)

    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_xlabel("Method")
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _row_label(row: dict[str, Any]) -> str:
    dataset = row.get("dataset", "unknown")
    method = row.get("method_family_label") or row.get("method_family") or row.get("method")
    return f"{method}\n{dataset}"
