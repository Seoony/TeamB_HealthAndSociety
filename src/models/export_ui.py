"""Exporta JSON estático para el front Vue (carpeta hermana TeamB_ui).

    python -m src.models.export_ui

No es un backend: copia métricas y predicciones ya calculadas.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from src.config import config
from src.models import protocol as P

UI_DATA = config.BASE_DIR.parent / "TeamB_ui" / "public" / "data"
PROCESSED = config.BASE_DIR / "data" / "processed"


def _read(name: str) -> pl.DataFrame:
    path = PROCESSED / name
    if not path.exists():
        raise FileNotFoundError(f"Falta {path}. Corre evaluate/robustness antes.")
    return pl.read_parquet(path)


def _stringify(df: pl.DataFrame) -> list[dict]:
    out = df
    for col, dtype in df.schema.items():
        if dtype in (pl.Date, pl.Datetime):
            out = out.with_columns(pl.col(col).cast(pl.String))
        elif dtype in (pl.Float32, pl.Float64):
            out = out.with_columns(pl.col(col).round(4))
    return out.to_dicts()


def _dump(name: str, payload: object) -> None:
    path = UI_DATA / name
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"  {path}  ({path.stat().st_size / 1024:.1f} KB)")


def export() -> Path:
    UI_DATA.mkdir(parents=True, exist_ok=True)
    preds = _read(P.PHASE3_PRED_NAME)
    metrics = _read(P.PHASE3_METRICS_NAME)
    intervals = _read(P.PHASE4_INTERVALS_NAME)
    coverage = _read(P.PHASE4_COVERAGE_NAME)
    sensitivity = _read(P.PHASE4_SENSITIVITY_NAME)
    transfer = _read(P.PHASE4_TRANSFER_NAME)

    departments = sorted(preds.get_column("department").unique().to_list())
    _dump(
        "meta.json",
        {
            "title": "IRAs 60+: pronóstico de hospitalizaciones a 4 semanas",
            "team": "Equipo B — UNSA / University of Oklahoma",
            "question": (
                "¿Cuántas hospitalizaciones en adultos de 60+ por IRA "
                "habrá en un departamento en las próximas 4 semanas?"
            ),
            "target": P.TARGET_COL,
            "horizon_weeks": P.HORIZON_WEEKS,
            "train": f"{P.YEAR_STUDY_START}–{P.YEAR_TRAIN_END}",
            "val": str(P.YEAR_VAL),
            "test": f"{P.YEAR_TEST[0]}–{P.YEAR_TEST[1]}",
            "covid_holdout": list(P.YEAR_COVID),
            "winter_weeks": list(P.WINTER_WEEKS),
            "pilots": list(P.PILOT_DEPARTMENTS),
            "departments": departments,
            "models": [
                {"id": "naive", "label": "Naive (persistencia)", "col": "yhat_naive"},
                {"id": "seasonal", "label": "Naive estacional", "col": "yhat_seasonal"},
                {"id": "sarima", "label": "ARIMA(1,1,1)", "col": "yhat_sarima"},
                {"id": "xgb", "label": "XGBoost", "col": "yhat_xgb"},
            ],
            "headline": (
                "ARIMA local gana al naive en test 2022–23 (+15% skill). "
                "XGBoost ayuda en Lima y empeora en Tumbes. "
                "El modelo de Arequipa no transfiere a otras regiones."
            ),
        },
    )
    _dump("metrics.json", _stringify(metrics))
    _dump(
        "series.json",
        _stringify(
            preds.select(
                "department",
                "week_start",
                "target_week_start",
                "target_ano",
                "target_semana",
                "split",
                "y",
                "yhat_naive",
                "yhat_seasonal",
                "yhat_sarima",
                "yhat_xgb",
            )
        ),
    )
    _dump(
        "intervals.json",
        _stringify(
            intervals.select(
                "department",
                "target_week_start",
                "model",
                "yhat",
                "lo",
                "hi",
                "covered",
            )
        ),
    )
    _dump("coverage.json", _stringify(coverage))
    _dump("sensitivity.json", _stringify(sensitivity))
    _dump("transfer.json", _stringify(transfer))
    return UI_DATA


def main() -> None:
    print(f"export → {UI_DATA}")
    export()
    print("listo. En TeamB_ui: npm install && npm run dev")


if __name__ == "__main__":
    main()
