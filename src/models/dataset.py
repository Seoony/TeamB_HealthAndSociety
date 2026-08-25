"""Construye la tabla de modelado (Fase 2). Sin entrenar, sin reportes.

    python -m src.models.dataset
    python -m src.models.dataset --department AREQUIPA
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

from src.config import config
from src.integrity.cleaning import load_cleaned
from src.models import protocol as P

PROCESSED_DIR = config.BASE_DIR / "data" / "processed"
POP_PATH = config.BASE_DIR / "data" / "raw" / "population_dept_long.parquet"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def modeling_path() -> Path:
    return PROCESSED_DIR / P.OUTPUT_NAME


def load_modeling(lazy: bool = False) -> pl.DataFrame | pl.LazyFrame:
    path = modeling_path()
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Corre: python -m src.models.dataset"
        )
    return pl.scan_parquet(path) if lazy else pl.read_parquet(path)


def _monday_panel(cleaned: pl.DataFrame) -> pl.DataFrame:
    """Panel semanal (lunes). Inserta semanas faltantes; no imputa métricas."""
    observed = cleaned.filter(pl.col("week_start").dt.weekday() == 1)
    dropped = cleaned.height - observed.height
    if dropped:
        log(f"  descartadas {dropped} filas con week_start que no es lunes")

    bounds = observed.group_by("department").agg(
        pl.col("week_start").min().alias("_wmin"),
        pl.col("week_start").max().alias("_wmax"),
        pl.col("ubigeo").first(),
        pl.col("geo_level").first(),
    )
    grid = bounds.with_columns(
        week_start=pl.date_ranges(pl.col("_wmin"), pl.col("_wmax"), "1w")
    ).drop("_wmin", "_wmax").explode("week_start")

    keep = [
        c for c in observed.columns
        if c not in {"department", "week_start", "ubigeo", "geo_level"}
    ]
    panel = grid.join(
        observed.select(["department", "week_start", *keep]),
        on=["department", "week_start"],
        how="left",
    )
    return panel.with_columns(
        flag_calendar_gap=pl.col(P.TARGET_COL).is_null(),
        ano=pl.coalesce(pl.col("ano"), pl.col("week_start").dt.iso_year().cast(pl.Int64)),
        semana=pl.coalesce(pl.col("semana"), pl.col("week_start").dt.week().cast(pl.Int64)),
        *[
            pl.col(c).fill_null(False)
            for c in ("flag_silence", "flag_death_no_hosp", "flag_hosp_corrected")
        ],
    )


def _join_population(panel: pl.DataFrame) -> pl.DataFrame:
    """Población total del depto-año (no es 60+). 2023 usa 2022."""
    if not POP_PATH.exists():
        log("  sin population_dept_long; population queda nula")
        return panel.with_columns(
            pl.lit(None).cast(pl.Float64).alias("population"),
            pl.lit(False).alias("flag_pop_carried_forward"),
            pl.lit(None).cast(pl.Float64).alias("hosp_60plus_per_100k"),
        )

    pop = (
        pl.read_parquet(POP_PATH)
        .select(
            department=pl.col("department").str.strip_chars().str.to_uppercase(),
            year=pl.col("year"),
            population=pl.col("population"),
        )
        .group_by(["department", "year"])
        .agg(pl.col("population").max())
    )
    latest = (
        pop.sort("year")
        .group_by("department")
        .agg(
            pl.col("population").last().alias("population_latest"),
        )
    )
    out = (
        panel.join(pop, left_on=["department", "ano"], right_on=["department", "year"], how="left")
        .join(latest, on="department", how="left")
        .with_columns(
            flag_pop_carried_forward=(
                pl.col("population").is_null() & pl.col("population_latest").is_not_null()
            ),
            population=pl.coalesce("population", "population_latest"),
        )
        .drop("population_latest")
        .with_columns(
            hosp_60plus_per_100k=pl.when(
                pl.col("population").is_not_null() & (pl.col("population") > 0)
            )
            .then(pl.col(P.TARGET_COL) / pl.col("population") * 100_000)
            .otherwise(None)
        )
    )
    n_carry = int(out["flag_pop_carried_forward"].sum())
    log(f"  población unida; {n_carry:,} filas con pop de último año (p.ej. 2023←2022)")
    return out


def _add_forecast_columns(panel: pl.DataFrame) -> pl.DataFrame:
    """Lags, medias móviles y target. Todo por departamento, sin futuro en X."""
    frame = panel.sort(["department", "week_start"])
    lag_exprs = [
        pl.col(col).shift(lag).over("department").alias(f"{col}_lag{lag}")
        for col in P.LAG_SOURCES
        for lag in P.LAGS
    ]
    roll_exprs = [
        pl.col(P.TARGET_COL)
        .rolling_mean(window_size=w, min_samples=w)
        .over("department")
        .alias(f"{P.TARGET_COL}_ma{w}")
        for w in P.ROLLING_WINDOWS
    ]
    return frame.with_columns(
        *lag_exprs,
        *roll_exprs,
        pl.col(P.TARGET_COL).shift(-P.HORIZON_WEEKS).over("department").alias("y"),
        pl.col("ano").shift(-P.HORIZON_WEEKS).over("department").alias("target_ano"),
        pl.col("week_start")
        .shift(-P.HORIZON_WEEKS)
        .over("department")
        .alias("target_week_start"),
    )


def _assign_split(frame: pl.DataFrame) -> pl.DataFrame:
    """Split por año del target. Filas sin y o sin features → drop."""
    complete = pl.all_horizontal(pl.col(c).is_not_null() for c in P.REQUIRED_FOR_MODEL)
    origin_ok = pl.col("ano") >= P.YEAR_STUDY_START
    usable = complete & origin_ok & pl.col("target_ano").is_not_null()

    split = (
        pl.when(~usable)
        .then(pl.lit(P.SPLIT_DROP))
        .when(pl.col("target_ano").is_between(P.YEAR_STUDY_START, P.YEAR_TRAIN_END))
        .then(pl.lit(P.SPLIT_TRAIN))
        .when(pl.col("target_ano") == P.YEAR_VAL)
        .then(pl.lit(P.SPLIT_VAL))
        .when(pl.col("target_ano").is_in(list(P.YEAR_COVID)))
        .then(pl.lit(P.SPLIT_COVID))
        .when(pl.col("target_ano").is_in(list(P.YEAR_TEST)))
        .then(pl.lit(P.SPLIT_TEST))
        .otherwise(pl.lit(P.SPLIT_DROP))
    )
    return frame.with_columns(split=split, usable=usable)


def _column_order(frame: pl.DataFrame) -> list[str]:
    front = [
        "geo_level",
        "department",
        "ubigeo",
        "week_start",
        "ano",
        "semana",
        "target_week_start",
        "target_ano",
        "y",
        "split",
        "usable",
    ]
    rest = [c for c in frame.columns if c not in front]
    return [c for c in front if c in frame.columns] + rest


def build_modeling_frame() -> pl.DataFrame:
    cleaned = load_cleaned(P.SOURCE_TABLE)
    log(f"fuente {P.SOURCE_TABLE}: {cleaned.height:,} filas")
    panel = _monday_panel(cleaned)
    n_gaps = int(panel["flag_calendar_gap"].sum())
    log(f"  panel lunes: {panel.height:,} filas  ({n_gaps:,} semanas insertadas)")
    panel = _join_population(panel)
    panel = _add_forecast_columns(panel)
    panel = _assign_split(panel)
    return panel.select(_column_order(panel))


def verify(frame: pl.DataFrame, department: str = "AREQUIPA") -> None:
    """Chequeos de leakage y alineación de y. Falla si el contrato se rompe."""
    dept = department.strip().upper()
    sample = frame.filter(pl.col("department") == dept).sort("week_start")
    if sample.is_empty():
        raise ValueError(f"no hay filas para {dept}")

    aligned = sample.select(
        "week_start",
        "hosp_60plus",
        "y",
        "target_week_start",
        y_from_future=pl.col("hosp_60plus").shift(-P.HORIZON_WEEKS),
    ).filter(pl.col("y").is_not_null())
    mismatch = aligned.filter(
        (pl.col("y") - pl.col("y_from_future")).abs() > 1e-9
    )
    if mismatch.height:
        raise AssertionError(
            f"y no coincide con hosp_60plus(t+{P.HORIZON_WEEKS}) en {dept}: "
            f"{mismatch.height} filas"
        )

    train = frame.filter(pl.col("split") == P.SPLIT_TRAIN)
    if train.filter(pl.col("target_ano") > P.YEAR_TRAIN_END).height:
        raise AssertionError("train contiene targets posteriores a 2018")
    if train.filter(pl.col("ano") < P.YEAR_STUDY_START).height:
        raise AssertionError("train contiene orígenes anteriores a 2006")

    test = frame.filter(pl.col("split") == P.SPLIT_TEST)
    if test.filter(~pl.col("target_ano").is_in(list(P.YEAR_TEST))).height:
        raise AssertionError("test tiene target_ano fuera de 2022-2023")

    covid = frame.filter(pl.col("split") == P.SPLIT_COVID)
    if covid.filter(~pl.col("target_ano").is_in(list(P.YEAR_COVID))).height:
        raise AssertionError("holdout_covid mal etiquetado")

    key_dups = frame.select(["department", "week_start"]).n_unique()
    if key_dups != frame.height:
        raise AssertionError(
            f"claves duplicadas department×week_start: {frame.height} filas / {key_dups} únicas"
        )

    plus4 = sample.filter(pl.col("y").is_not_null()).select(
        delta_days=(pl.col("target_week_start") - pl.col("week_start")).dt.total_days()
    )
    bad_horizon = plus4.filter(pl.col("delta_days") != 7 * P.HORIZON_WEEKS)
    if bad_horizon.height:
        raise AssertionError(
            f"horizonte distinto de {P.HORIZON_WEEKS} semanas en {bad_horizon.height} filas"
        )

    counts = (
        frame.group_by("split")
        .len()
        .sort("split")
    )
    log("splits:")
    for row in counts.iter_rows(named=True):
        log(f"  {row['split']:<16s} {row['len']:>8,}")

    show = (
        sample.filter(pl.col("split") == P.SPLIT_TRAIN)
        .select(
            "week_start",
            "ano",
            "hosp_60plus",
            "y",
            "target_week_start",
            "target_ano",
            "split",
        )
        .head(8)
    )
    log(f"muestra {dept} (train):")
    print(show)

    log(
        f"ok: y = {P.TARGET_COL}(t+{P.HORIZON_WEEKS}); "
        f"train.target_ano ≤ {P.YEAR_TRAIN_END}; "
        f"test.target_ano ∈ {P.YEAR_TEST}"
    )


def run(department: str | None = None) -> Path:
    t0 = time.time()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frame = build_modeling_frame()
    out = modeling_path()
    frame.write_parquet(out, compression="zstd")
    log(f"escrito {out.relative_to(config.BASE_DIR)}  ({frame.height:,} filas)")
    verify(frame, department=department or "AREQUIPA")
    log(f"fase 2 lista en {time.time() - t0:.2f}s")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tabla de modelado: y = hosp_60plus(t+4)."
    )
    parser.add_argument(
        "--department",
        default="AREQUIPA",
        help="Departamento para la verificación impresa (default: AREQUIPA).",
    )
    args = parser.parse_args()
    run(department=args.department)


if __name__ == "__main__":
    main()
