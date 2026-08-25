"""Limpieza de integridad — corazón del pipeline.

Entrada : data/raw/*.parquet
Salida  : data/processed/<tabla>_cleaned.parquet

Una fila limpia = un lugar × una semana. No escribe reportes ni figuras.
R2 queda como `flag_silence` en la tabla (insumo de análisis), no como CSV.

    python -m src.integrity.cleaning
    python -m src.integrity.cleaning --table iras_weekly_dept
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

from src.config import config
from src.integrity.schema import (
    CLEANING_PLAN,
    FLAG_COLS,
    CleaningSpec,
    analysis_columns,
    required_raw_columns,
)

RAW_DIR = config.BASE_DIR / "data" / "raw"
PROCESSED_DIR = config.BASE_DIR / "data" / "processed"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def silence_mask() -> pl.Expr:
    """Casos reportados sin hospitalizaciones ni muertes (posible subregistro)."""
    return (
        (pl.col("cases_total").fill_null(0) > 0)
        & (pl.col("hosp_total").fill_null(0) == 0)
        & (pl.col("deaths_total").fill_null(0) == 0)
    )


def deduplicate(
    lf: pl.LazyFrame,
    key_cols: list[str],
    metric_cols: list[str],
    passthrough: list[str],
) -> pl.LazyFrame:
    """R1: una fila por clave; `max()` en métricas para no perder magnitud."""
    return (
        lf.group_by(key_cols, maintain_order=False)
        .agg(
            [pl.col(c).max() for c in metric_cols]
            + [pl.col(c).first() for c in passthrough]
        )
        .sort(key_cols)
    )


def add_analysis_flags(
    lf: pl.LazyFrame,
    cases_col: str = "cases_total",
) -> pl.LazyFrame:
    """R3/R4 + silencio a nivel de fila.

    - flag_death_no_hosp: no se elimina (posible muerte domiciliaria).
    - flag_hosp_corrected: clamp `cases = max(cases, hosp)`.
    - flag_silence: cases>0 y hosp=deaths=0 (subregistro / atención ambulatoria).
    """
    hosp = pl.col("hosp_total").fill_null(0)
    deaths = pl.col("deaths_total").fill_null(0)
    cases = pl.col(cases_col).fill_null(0)

    return lf.with_columns(
        (deaths > hosp).alias("flag_death_no_hosp"),
        (hosp > cases).alias("flag_hosp_corrected"),
        silence_mask().alias("flag_silence"),
    ).with_columns(
        pl.when(pl.col("flag_hosp_corrected"))
        .then(pl.col("hosp_total"))
        .otherwise(pl.col(cases_col))
        .alias(cases_col)
    )


def finalize_frame(df: pl.DataFrame, spec: CleaningSpec) -> pl.DataFrame:
    """Contrato estable: geo_level, orden de columnas, tipos de flags."""
    ordered = [c for c in analysis_columns(spec) if c == "geo_level" or c in df.columns]
    return (
        df.with_columns(
            pl.lit(spec["geo_level"]).alias("geo_level"),
            *[pl.col(c).cast(pl.Boolean) for c in FLAG_COLS],
        )
        .select(ordered)
    )


def clean_frame(lf: pl.LazyFrame, spec: CleaningSpec) -> pl.LazyFrame:
    """Transformación pura: raw scan → lazy limpio (sin I/O)."""
    cols = required_raw_columns(spec)
    return add_analysis_flags(
        deduplicate(
            lf.select(cols),
            spec["key"],
            spec["metrics"],
            spec["passthrough"],
        )
    )


def processed_path(name: str) -> Path:
    return PROCESSED_DIR / f"{name}_cleaned.parquet"


def load_cleaned(name: str, lazy: bool = False) -> pl.DataFrame | pl.LazyFrame:
    """Carga una tabla processed. Punto de entrada para análisis y modelos."""
    if name not in CLEANING_PLAN:
        known = ", ".join(CLEANING_PLAN)
        raise KeyError(f"tabla desconocida '{name}'. Usa: {known}")
    path = processed_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Corre: python -m src.integrity.cleaning --table {name}"
        )
    return pl.scan_parquet(path) if lazy else pl.read_parquet(path)


def clean_table(name: str, spec: CleaningSpec | None = None) -> dict:
    """Limpia una tabla y la escribe en processed/. Sin reportes."""
    spec = spec or CLEANING_PLAN[name]
    src = RAW_DIR / spec["raw"]
    if not src.exists():
        log(f"omitida '{name}': no está {src.relative_to(config.BASE_DIR)}")
        return {}

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    log(f"limpiando {name}")

    raw = pl.scan_parquet(src)
    n_in = raw.select(pl.len()).collect().item()
    df = finalize_frame(clean_frame(raw, spec).collect(), spec)

    n_out = df.height
    out = processed_path(name)
    df.write_parquet(out, compression="zstd")

    summary = {
        "name": name,
        "n_in": n_in,
        "n_out": n_out,
        "n_dropped": n_in - n_out,
        "n_silence": int(df["flag_silence"].sum()),
        "n_death_gt_hosp": int(df["flag_death_no_hosp"].sum()),
        "n_hosp_gt_cases": int(df["flag_hosp_corrected"].sum()),
        "path": out,
    }
    log(
        f"  {n_in:,} → {n_out:,}  (−{summary['n_dropped']:,} dup)  "
        f"silence={summary['n_silence']:,}  "
        f"death>hosp={summary['n_death_gt_hosp']:,}  "
        f"hosp>cases={summary['n_hosp_gt_cases']:,}  "
        f"→ {out.relative_to(config.BASE_DIR)}"
    )
    return summary


def run_clean(tables: list[str] | None = None) -> list[dict]:
    """Limpia todas las tablas del contrato, o un subconjunto (`--table`)."""
    names = tables or list(CLEANING_PLAN)
    unknown = [n for n in names if n not in CLEANING_PLAN]
    if unknown:
        known = ", ".join(CLEANING_PLAN)
        raise KeyError(f"tablas desconocidas {unknown}. Usa: {known}")

    log(f"polars {pl.__version__}  threads={pl.thread_pool_size()}")
    started = time.time()
    results = [clean_table(name) for name in names]
    done = [r for r in results if r]
    log(f"listo: {len(done)}/{len(names)} tablas en {time.time() - started:.2f}s")
    return done


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Limpia tablas raw → processed (sin reportes)."
    )
    parser.add_argument(
        "--table",
        action="append",
        choices=list(CLEANING_PLAN),
        help="Limpia solo esta tabla. Se puede repetir. Por defecto: todas.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_clean(args.table)


if __name__ == "__main__":
    main()
