"""Fase 3: evaluación walk-forward (naive, estacional, SARIMA, XGBoost).

    python -m src.models.evaluate

Salidas (parquet, no PDF):
    data/processed/phase3_predictions.parquet
    data/processed/phase3_metrics.parquet
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="No supported index")

import numpy as np
import polars as pl

from src.config import config
from src.models import protocol as P
from src.models.dataset import load_modeling
from src.models.forecasts import (
    add_naive_forecasts,
    walk_forward_sarima,
    walk_forward_xgb,
)

PROCESSED_DIR = config.BASE_DIR / "data" / "processed"
MODELS = ("naive", "seasonal", "sarima", "xgb")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def predictions_path() -> Path:
    return PROCESSED_DIR / P.PHASE3_PRED_NAME


def metrics_path() -> Path:
    return PROCESSED_DIR / P.PHASE3_METRICS_NAME


def _winter_mask(frame: pl.DataFrame) -> pl.Series:
    lo, hi = P.WINTER_WEEKS
    return frame.get_column("target_week_start").dt.week().is_between(lo, hi)


def _scores(y: np.ndarray, yhat: np.ndarray) -> tuple[float, float]:
    err = yhat - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return mae, rmse


def _metric_rows(frame: pl.DataFrame, split: str, scope: str, department: str | None) -> list[dict]:
    y = frame["y"].to_numpy()
    naive_mae, naive_rmse = _scores(y, frame["yhat_naive"].to_numpy())
    rows = []
    for model in MODELS:
        col = f"yhat_{model}"
        mae, rmse = _scores(y, frame[col].to_numpy())
        rows.append({
            "split": split,
            "scope": scope,
            "department": department,
            "model": model,
            "n": frame.height,
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mae_skill_vs_naive": round(1.0 - mae / naive_mae, 4) if naive_mae else None,
            "rmse_skill_vs_naive": round(1.0 - rmse / naive_rmse, 4) if naive_rmse else None,
        })
    return rows


def compute_metrics(preds: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict] = []
    for split in (P.SPLIT_VAL, P.SPLIT_TEST):
        part = preds.filter(pl.col("split") == split)
        if part.is_empty():
            continue
        rows.extend(_metric_rows(part, split, "overall", None))
        winter = part.filter(_winter_mask(part))
        if winter.height:
            rows.extend(_metric_rows(winter, split, "winter", None))
        for dept in part.get_column("department").unique().sort().to_list():
            dept_df = part.filter(pl.col("department") == dept)
            rows.extend(_metric_rows(dept_df, split, "department", dept))
    return pl.DataFrame(rows)


def _run_split(frame: pl.DataFrame, split: str, history: tuple[str, ...]) -> pl.DataFrame:
    log(f"{split}: naive / estacional")
    base = add_naive_forecasts(frame).filter(pl.col("split") == split)
    log(f"{split}: XGBoost walk-forward (refit cada {P.REFIT_EVERY_WEEKS} orígenes)")
    xgb = walk_forward_xgb(frame, split, history)
    log(f"{split}: SARIMA walk-forward por departamento")
    sarima = walk_forward_sarima(frame, split)
    return (
        base.join(
            xgb.select(["department", "week_start", "yhat_xgb"]),
            on=["department", "week_start"],
            how="left",
        )
        .join(
            sarima.select(["department", "week_start", "yhat_sarima"]),
            on=["department", "week_start"],
            how="left",
        )
        .with_columns(
            target_semana=pl.col("target_week_start").dt.week(),
        )
        .select(
            "department",
            "week_start",
            "ano",
            "semana",
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
    )


def evaluate() -> tuple[pl.DataFrame, pl.DataFrame]:
    frame = load_modeling()
    usable = frame.filter(pl.col("split") != P.SPLIT_DROP)
    log(f"modeling: {usable.height:,} filas usables")

    val = _run_split(usable, P.SPLIT_VAL, (P.SPLIT_TRAIN,))
    test = _run_split(usable, P.SPLIT_TEST, (P.SPLIT_TRAIN, P.SPLIT_VAL))
    preds = pl.concat([val, test], how="vertical")
    metrics = compute_metrics(preds)
    return preds, metrics


def _print_summary(metrics: pl.DataFrame) -> None:
    show = metrics.filter(pl.col("scope").is_in(["overall", "winter"])).sort(
        ["split", "scope", "model"]
    )
    print(show)
    winners = (
        metrics.filter(pl.col("scope") == "overall")
        .sort(["split", "mae"])
        .group_by("split", maintain_order=True)
        .first()
        .select("split", "model", "mae", "rmse", "mae_skill_vs_naive")
    )
    log("mejor MAE overall por split:")
    print(winners)
    pilots = metrics.filter(
        (pl.col("scope") == "department")
        & (pl.col("department").is_in(list(P.PILOT_DEPARTMENTS)))
        & (pl.col("split") == P.SPLIT_TEST)
    ).sort(["department", "mae"])
    log("test por departamento piloto (ordenado por MAE):")
    print(pilots.select("department", "model", "n", "mae", "rmse", "mae_skill_vs_naive"))


def main() -> None:
    t0 = time.time()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    preds, metrics = evaluate()
    predictions_path().parent.mkdir(parents=True, exist_ok=True)
    preds.write_parquet(predictions_path(), compression="zstd")
    metrics.write_parquet(metrics_path(), compression="zstd")
    log(f"predicciones → {predictions_path().relative_to(config.BASE_DIR)}")
    log(f"métricas     → {metrics_path().relative_to(config.BASE_DIR)}")
    _print_summary(metrics)
    log(f"fase 3 lista en {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
