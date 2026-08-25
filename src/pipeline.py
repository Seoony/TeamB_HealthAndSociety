"""Orquestador del corazón: solo limpieza.

    python -m src.pipeline
    python -m src.pipeline --table iras_weekly_dept --table pneumonia_dept

Los reportes (validation / outliers / analysis) no se invocan aquí.
"""

from __future__ import annotations

import argparse

from src.integrity.cleaning import run_clean
from src.integrity.schema import CLEANING_PLAN


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline núcleo: raw → processed."
    )
    parser.add_argument(
        "--table",
        action="append",
        choices=list(CLEANING_PLAN),
        help="Limpia solo estas tablas. Por defecto: las 4 del contrato.",
    )
    args = parser.parse_args()
    run_clean(args.table)


if __name__ == "__main__":
    main()
