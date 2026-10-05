"""Command-line interface: ``calibkit report predictions.csv``.

The CSV must have two columns: ``y_true`` (0/1) and ``y_prob`` (probability of class 1).
"""

from __future__ import annotations

import argparse
import csv
import sys

from .metrics import brier_score, expected_calibration_error, max_calibration_error
from .scaling import TemperatureScaler


def _read_csv(path: str) -> tuple[list[int], list[float]]:
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or not {"y_true", "y_prob"} <= set(reader.fieldnames):
            raise SystemExit("error: CSV must contain columns 'y_true' and 'y_prob'")
        y_true, y_prob = [], []
        for row in reader:
            y_true.append(int(row["y_true"]))
            y_prob.append(float(row["y_prob"]))
    return y_true, y_prob


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="calibkit", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    rep = sub.add_parser(
        "report", help="print calibration metrics before/after temperature scaling"
    )
    rep.add_argument("csv_path")
    rep.add_argument("--bins", type=int, default=10)
    args = parser.parse_args(argv)

    y_true, y_prob = _read_csv(args.csv_path)
    scaler = TemperatureScaler()
    scaled = scaler.fit_transform(y_true, y_prob)

    print(f"samples: {len(y_true)}")
    print(f"temperature: {scaler.temperature_:.3f}")
    print(f"{'metric':<8}{'before':>10}{'after':>10}")
    rows = [
        ("ECE", expected_calibration_error(y_true, y_prob, args.bins),
         expected_calibration_error(y_true, scaled, args.bins)),
        ("MCE", max_calibration_error(y_true, y_prob, args.bins),
         max_calibration_error(y_true, scaled, args.bins)),
        ("Brier", brier_score(y_true, y_prob), brier_score(y_true, scaled)),
    ]
    for name, before, after in rows:
        print(f"{name:<8}{before:>10.4f}{after:>10.4f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
