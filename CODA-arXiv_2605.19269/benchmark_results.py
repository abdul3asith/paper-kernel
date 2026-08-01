"""Utilities for recording SGEMM benchmark results in one CSV file."""

import csv
from pathlib import Path
from typing import Any


RESULTS_CSV = Path(__file__).resolve().with_name("Results.csv")
FIELDNAMES = [
    "implementation",
    "device",
    "m",
    "k",
    "n",
    "repeats",
    "average_ms",
    "tflops",
    "c00",
]


def append_result(**result: Any) -> Path:
    """Append a benchmark row and return the absolute results path."""
    write_header = not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0
    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({field: result.get(field, "") for field in FIELDNAMES})
    return RESULTS_CSV
