"""Fase 4: intervalos, sensibilidad y transferencia externa.

    python -m src.models.robustness

No reentrena SARIMA (usa Fase 3). Sí reentrena XGBoost en las pruebas
COVID / sin flag_silence / modelo Arequipa → otros departamentos.
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl

from src.config import config
from src.models import protocol as P
from src.models.dataset import load_modeling
from src.models.forecasts import add_naive_forecasts, walk_forward_xgb

PROCESSED_DIR = config.BASE_DIR / "data" / "processed"
MODELS = ("naive", "seasonal", "sarima", "xgb")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _scores(y: np.ndarray, yhat: np.ndarray) -> tuple[float, float]:
    err = yhat - y
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2)))


def load_phase3_predictions() -> pl.DataFrame:
    path = PROCESSED_DIR / P.PHASE3_PRED_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Corre primero: python -m src.models.evaluate"
        )
    return pl.read_parquet(path)


# --------------------------------------------------------------------------- #
# Intervalos conformales (90%) calibrados en val, evaluados en test
# --------------------------------------------------------------------------- #
def _residual_bounds(residuals: np.ndarray) -> tuple[float, float]:
    tail = 100.0 * P.INTERVAL_ALPHA / 2.0
    lo = float(np.quantile(residuals, tail / 100.0))
    hi = float(np.quantile(residuals, 1.0 - tail / 100.0))
    return lo, hi


def conformal_intervals(preds: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    val = preds.filter(pl.col("split") == P.SPLIT_VAL)
    test = preds.filter(pl.col("split") == P.SPLIT_TEST)
    pieces: list[pl.DataFrame] = []
    coverage_rows: list[dict] = []

    pooled: dict[str, tuple[float, float]] = {}
    for model in MODELS:
        col = f"yhat_{model}"
        pooled[model] = _residual_bounds(
            (val["y"] - val[col]).to_numpy()
        )

    dept_bounds: dict[tuple[str, str], tuple[float, float]] = {}
    for dept in test.get_column("department").unique().to_list():
        val_d = val.filter(pl.col("department") == dept)
        if val_d.height < 30:
            continue
        for model in MODELS:
            col = f"yhat_{model}"
            dept_bounds[(dept, model)] = _residual_bounds(
                (val_d["y"] - val_d[col]).to_numpy()
            )

    for model in MODELS:
        col = f"yhat_{model}"
        q_lo_p, q_hi_p = pooled[model]
        part = test.select(
            "department",
            "week_start",
            "target_week_start",
            "y",
            yhat=pl.col(col),
        ).with_columns(pl.lit(model).alias("model"))

        bound_rows = []
        for dept in part.get_column("department").unique().to_list():
            lo, hi = dept_bounds.get((dept, model), (q_lo_p, q_hi_p))
            method = "per_department" if (dept, model) in dept_bounds else "pooled"
            if (hi - lo) < 0.5:
                lo, hi = q_lo_p, q_hi_p
                method = "pooled_local_too_narrow"
            bound_rows.append({"department": dept, "q_lo": lo, "q_hi": hi, "method": method})
        bounds = pl.DataFrame(bound_rows)
        part = (
            part.join(bounds, on="department", how="left")
            .with_columns(
                lo=(pl.col("yhat") + pl.col("q_lo")).clip(lower_bound=0.0),
                hi=pl.col("yhat") + pl.col("q_hi"),
            )
            .with_columns(
                hi=pl.max_horizontal(pl.col("hi"), pl.col("lo") + 1.0),
            )
            .with_columns(
                covered=(pl.col("y") >= pl.col("lo")) & (pl.col("y") <= pl.col("hi")),
                width=pl.col("hi") - pl.col("lo"),
            )
        )
        pieces.append(part.select(
            "department", "week_start", "target_week_start", "model",
            "y", "yhat", "lo", "hi", "covered", "width", "method",
        ))

        for scope, sub in (("overall", part),):
            coverage_rows.append({
                "split": P.SPLIT_TEST,
                "scope": scope,
                "department": None,
                "model": model,
                "level": 1.0 - P.INTERVAL_ALPHA,
                "n": sub.height,
                "coverage": round(float(sub["covered"].mean()), 4),
                "mean_width": round(float(sub["width"].mean()), 4),
                "method": "per_department_fallback_pooled",
            })
        for dept in P.PILOT_DEPARTMENTS:
            sub = part.filter(pl.col("department") == dept)
            if sub.is_empty():
                continue
            coverage_rows.append({
                "split": P.SPLIT_TEST,
                "scope": "department",
                "department": dept,
                "model": model,
                "level": 1.0 - P.INTERVAL_ALPHA,
                "n": sub.height,
                "coverage": round(float(sub["covered"].mean()), 4),
                "mean_width": round(float(sub["width"].mean()), 4),
                "method": sub["method"][0],
            })

    intervals = pl.concat(pieces, how="vertical")
    return intervals, pl.DataFrame(coverage_rows)


# --------------------------------------------------------------------------- #
# Sensibilidad
# --------------------------------------------------------------------------- #
def _row(experiment: str, model: str, y: np.ndarray, yhat: np.ndarray, note: str) -> dict:
    mae, rmse = _scores(y, yhat)
    return {
        "experiment": experiment,
        "model": model,
        "n": int(y.size),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "note": note,
    }


def sensitivity(preds: pl.DataFrame, modeling: pl.DataFrame) -> pl.DataFrame:
    test = preds.filter(pl.col("split") == P.SPLIT_TEST)
    flags = modeling.select(
        "department",
        "week_start",
        "flag_silence",
        "flag_calendar_gap",
        "flag_death_no_hosp",
    )
    joined = test.join(flags, on=["department", "week_start"], how="left")
    rows: list[dict] = []

    y_all = joined["y"].to_numpy()
    for model in MODELS:
        rows.append(_row(
            "baseline_test", model,
            y_all, joined[f"yhat_{model}"].to_numpy(),
            "test 2022-23, todas las filas",
        ))

    clear = joined.filter(~pl.col("flag_silence"))
    for model in MODELS:
        rows.append(_row(
            "exclude_silence", model,
            clear["y"].to_numpy(), clear[f"yhat_{model}"].to_numpy(),
            "test sin filas flag_silence",
        ))

    no_gap = joined.filter(~pl.col("flag_calendar_gap"))
    for model in MODELS:
        rows.append(_row(
            "exclude_calendar_gap", model,
            no_gap["y"].to_numpy(), no_gap[f"yhat_{model}"].to_numpy(),
            "test sin semanas insertadas",
        ))

    log("sensibilidad: XGB entrenando también con COVID")
    xgb_covid = walk_forward_xgb(
        modeling.filter(pl.col("split") != P.SPLIT_DROP),
        P.SPLIT_TEST,
        (P.SPLIT_TRAIN, P.SPLIT_VAL, P.SPLIT_COVID),
        yhat_name="yhat_xgb_covid",
    )
    covid_join = test.join(
        xgb_covid.select(["department", "week_start", "yhat_xgb_covid"]),
        on=["department", "week_start"],
        how="left",
    )
    rows.append(_row(
        "xgb_train_with_covid", "xgb",
        covid_join["y"].to_numpy(), covid_join["yhat_xgb_covid"].to_numpy(),
        "XGB walk-forward con holdout_covid en el historial",
    ))

    log("sensibilidad: XGB sin flag_silence")
    no_silence_feats = tuple(c for c in P.XGB_FEATURES if c != "flag_silence")
    xgb_ns = walk_forward_xgb(
        modeling.filter(pl.col("split") != P.SPLIT_DROP),
        P.SPLIT_TEST,
        (P.SPLIT_TRAIN, P.SPLIT_VAL),
        feature_cols=no_silence_feats,
        yhat_name="yhat_xgb_nofilence",
    )
    ns_join = test.join(
        xgb_ns.select(["department", "week_start", "yhat_xgb_nofilence"]),
        on=["department", "week_start"],
        how="left",
    )
    rows.append(_row(
        "xgb_without_silence_flag", "xgb",
        ns_join["y"].to_numpy(), ns_join["yhat_xgb_nofilence"].to_numpy(),
        "misma walk-forward, feature flag_silence quitada",
    ))

    log("sensibilidad: estrés COVID (predecir 2020-21 con train 2006-18)")
    usable = modeling.filter(pl.col("split") != P.SPLIT_DROP)
    covid_base = add_naive_forecasts(usable).filter(pl.col("split") == P.SPLIT_COVID)
    xgb_on_covid = walk_forward_xgb(
        usable, P.SPLIT_COVID, (P.SPLIT_TRAIN,), yhat_name="yhat_xgb"
    )
    covid_eval = covid_base.join(
        xgb_on_covid.select(["department", "week_start", "yhat_xgb"]),
        on=["department", "week_start"],
        how="left",
    )
    for model, col in (("naive", "yhat_naive"), ("seasonal", "yhat_seasonal"), ("xgb", "yhat_xgb")):
        rows.append(_row(
            "stress_covid_2020_21", model,
            covid_eval["y"].to_numpy(), covid_eval[col].to_numpy(),
            "evaluar 2020-21; XGB solo vio train 2006-18",
        ))

    # skill vs naive of the same experiment
    by_exp: dict[str, float] = {}
    for r in rows:
        if r["model"] == "naive":
            by_exp[r["experiment"]] = r["mae"]
    baseline_naive = by_exp.get("baseline_test")
    for r in rows:
        base = by_exp.get(r["experiment"]) or baseline_naive
        r["mae_skill_vs_naive"] = (
            round(1.0 - r["mae"] / base, 4) if base else None
        )

    return pl.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Transferencia Arequipa → otros pilotos
# --------------------------------------------------------------------------- #
def transfer(preds: pl.DataFrame, modeling: pl.DataFrame) -> pl.DataFrame:
    source = P.TRANSFER_SOURCE
    targets = [d for d in P.PILOT_DEPARTMENTS if d != source]
    log(f"transferencia: modelo {source} (sin feature department) → {targets}")
    usable = modeling.filter(pl.col("split") != P.SPLIT_DROP)
    xfer = walk_forward_xgb(
        usable,
        P.SPLIT_TEST,
        (P.SPLIT_TRAIN, P.SPLIT_VAL),
        feature_cols=P.TRANSFER_FEATURES,
        train_departments=[source],
        pred_departments=targets,
        yhat_name="yhat_xgb_transfer",
    )
    test = preds.filter(
        (pl.col("split") == P.SPLIT_TEST)
        & (pl.col("department").is_in(targets))
    )
    joined = test.join(
        xfer.select(["department", "week_start", "yhat_xgb_transfer"]),
        on=["department", "week_start"],
        how="left",
    )

    rows = []
    for dept in targets:
        part = joined.filter(pl.col("department") == dept)
        y = part["y"].to_numpy()
        naive_mae, _ = _scores(y, part["yhat_naive"].to_numpy())
        local_xgb_mae, _ = _scores(y, part["yhat_xgb"].to_numpy())
        tr_mae, tr_rmse = _scores(y, part["yhat_xgb_transfer"].to_numpy())
        rows.append({
            "source": source,
            "target": dept,
            "n": part.height,
            "mae_transfer": round(tr_mae, 4),
            "rmse_transfer": round(tr_rmse, 4),
            "mae_local_naive": round(naive_mae, 4),
            "mae_local_xgb": round(local_xgb_mae, 4),
            "skill_vs_local_naive": round(1.0 - tr_mae / naive_mae, 4) if naive_mae else None,
            "skill_vs_local_xgb": round(1.0 - tr_mae / local_xgb_mae, 4) if local_xgb_mae else None,
        })
    return pl.DataFrame(rows)


def _print_summary(
    coverage: pl.DataFrame,
    sensitivity_df: pl.DataFrame,
    transfer_df: pl.DataFrame,
) -> None:
    log("cobertura intervalos 90% (test):")
    print(
        coverage.filter(pl.col("scope") == "overall").select(
            "model", "n", "coverage", "mean_width", "level"
        )
    )
    log("pilotos (cobertura):")
    print(
        coverage.filter(pl.col("scope") == "department")
        .select("department", "model", "coverage", "mean_width")
        .sort(["department", "model"])
    )
    log("sensibilidad:")
    print(
        sensitivity_df.select(
            "experiment", "model", "n", "mae", "mae_skill_vs_naive", "note"
        )
    )
    log("transferencia Arequipa → otros:")
    print(transfer_df)


def main() -> None:
    t0 = time.time()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    preds = load_phase3_predictions()
    modeling = load_modeling()
    log(f"fase 3 preds: {preds.height:,}  modeling: {modeling.height:,}")

    log("intervalos conformales 90%")
    intervals, coverage = conformal_intervals(preds)
    sens = sensitivity(preds, modeling)
    xfer = transfer(preds, modeling)

    intervals.write_parquet(PROCESSED_DIR / P.PHASE4_INTERVALS_NAME, compression="zstd")
    coverage.write_parquet(PROCESSED_DIR / P.PHASE4_COVERAGE_NAME, compression="zstd")
    sens.write_parquet(PROCESSED_DIR / P.PHASE4_SENSITIVITY_NAME, compression="zstd")
    xfer.write_parquet(PROCESSED_DIR / P.PHASE4_TRANSFER_NAME, compression="zstd")

    log(f"intervalos   → data/processed/{P.PHASE4_INTERVALS_NAME}")
    log(f"cobertura    → data/processed/{P.PHASE4_COVERAGE_NAME}")
    log(f"sensibilidad → data/processed/{P.PHASE4_SENSITIVITY_NAME}")
    log(f"transfer     → data/processed/{P.PHASE4_TRANSFER_NAME}")
    _print_summary(coverage, sens, xfer)
    log(f"fase 4 lista en {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
