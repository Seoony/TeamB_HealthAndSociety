"""
Limpieza y Correcciones de Integridad
=====================================

Proyecto : UNSA & Oklahoma - Team B (Salud y Sociedad)
Modulo   : src.integrity.cleaning
Ejecutar : python -X utf8 -m src.integrity.cleaning
Entradas : data/raw/*.parquet
Salidas  : data/processed/*_cleaned.parquet
           reports/integrity/cleaning_report.txt
           reports/integrity/silence_by_province.csv

Aplica las cuatro recomendaciones derivadas de `validation.py`:

R1. Deduplicacion de tablas weekly
    Las tablas `iras_weekly_*` y `pneumonia_*` repiten cada
    `(department, week_start)` por sub-ubigeo provincial. Colapsamos a
    una fila por clave natural tomando `max()` (preserva la magnitud).

R2. Investigar silencios al 25% en datos provinciales
    Generamos `silence_by_province.csv`: para cada provincia, % de
    semanas con `cases > 0` pero `hosp = 0 AND deaths = 0`. Incluye
    tendencia temporal (pendiente por OLS simple del % anual).

R3. Marcar `deaths > hosp` como flag (NO eliminar)
    Agregamos `flag_death_no_hosp = True` cuando `deaths_total >
    hosp_total`. Razon: pueden ser muertes domiciliarias legitimas y
    descartarlas sesgaria el modelo de mortalidad.

R4. Corregir `hosp > cases` (clamp con flag)
    Cuando `hosp_total > cases_total` ajustamos `cases_total =
    hosp_total` (no se puede hospitalizar sin caso => caso al menos
    igual). Tambien agregamos `flag_hosp_corrected = True` para auditar.

El modulo NO toca `data/raw/`. Genera ficheros nuevos en
`data/processed/` con el sufijo `_cleaned`.
"""

from __future__ import annotations

import time
from pathlib import Path

import polars as pl

# --------------------------------------------------------------------------- #
# Configuracion
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports" / "integrity"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Pasos de limpieza individuales (cada uno devuelve un LazyFrame)
# --------------------------------------------------------------------------- #
def deduplicate(lf: pl.LazyFrame, key_cols: list[str],
                metric_cols: list[str], passthrough: list[str]
                ) -> pl.LazyFrame:
    """R1: Colapsa filas duplicadas por `key_cols` con `max()` en metricas
    numericas y `first()` en columnas textuales/clave secundarias."""
    return (
        lf.group_by(key_cols, maintain_order=False)
        .agg([pl.col(c).max() for c in metric_cols]
             + [pl.col(c).first() for c in passthrough])
        .sort(key_cols)
    )


def flag_death_gt_hosp(lf: pl.LazyFrame) -> pl.LazyFrame:
    """R3: marca filas con `deaths_total > hosp_total` (no las elimina)."""
    return lf.with_columns(
        (pl.col("deaths_total").fill_null(0) > pl.col("hosp_total").fill_null(0))
        .alias("flag_death_no_hosp")
    )


def clamp_hosp_gt_cases(lf: pl.LazyFrame, cases_col: str = "cases_total"
                        ) -> pl.LazyFrame:
    """R4: cuando `hosp_total > cases`, fuerza `cases = hosp_total` y deja
    una bandera `flag_hosp_corrected` para auditoria."""
    return lf.with_columns(
        (pl.col("hosp_total").fill_null(0) > pl.col(cases_col).fill_null(0))
        .alias("flag_hosp_corrected")
    ).with_columns(
        pl.when(pl.col("flag_hosp_corrected"))
        .then(pl.col("hosp_total"))
        .otherwise(pl.col(cases_col))
        .alias(cases_col)
    )


# --------------------------------------------------------------------------- #
# R2: Reporte de silencios por provincia + tendencia temporal
# --------------------------------------------------------------------------- #
def silence_by_province(prov_path: Path, year_col: str = "ano") -> pl.DataFrame:
    """R2: para cada provincia calcula el % de semanas con casos>0 y sin
    hospitalizaciones ni decesos. Incluye:

    - `pct_silence` : porcentaje global de silencios en la serie historica.
    - `slope_pct_per_year`: pendiente lineal del % anual (si > 0 el sub-
      registro empeora; si < 0 mejora). Si solo hay un anio devuelve 0.
    - `n_obs` : cantidad de semanas observadas.
    """
    lf = pl.scan_parquet(prov_path)

    is_silence = (
        (pl.col("cases_total").fill_null(0) > 0)
        & (pl.col("hosp_total").fill_null(0) == 0)
        & (pl.col("deaths_total").fill_null(0) == 0)
    ).alias("_silence")

    # % global por provincia
    global_pct = (
        lf.with_columns(is_silence)
        .group_by(["department", "province"])
        .agg(
            pct_silence=(pl.col("_silence").sum() / pl.len() * 100).round(3),
            n_obs=pl.len(),
        )
    )

    # % anual por provincia
    yearly = (
        lf.with_columns(is_silence)
        .group_by(["department", "province", year_col])
        .agg(pct=(pl.col("_silence").sum() / pl.len() * 100))
        .collect()
    )

    # Pendiente OLS simple: slope = cov(x,y)/var(x)
    def _slope(sub: pl.DataFrame) -> float:
        if sub.height < 2:
            return 0.0
        x = sub.get_column(year_col).to_numpy().astype(float)
        y = sub.get_column("pct").to_numpy().astype(float)
        x_mean = x.mean(); y_mean = y.mean()
        denom = ((x - x_mean) ** 2).sum()
        if denom == 0:
            return 0.0
        return float(((x - x_mean) * (y - y_mean)).sum() / denom)

    slopes = (
        yearly.group_by(["department", "province"], maintain_order=True)
        .agg(pl.struct([year_col, "pct"]).alias("_records"))
        .with_columns(
            slope_pct_per_year=pl.col("_records").map_elements(
                lambda recs: _slope(pl.DataFrame(recs.to_list())),
                return_dtype=pl.Float64,
            ).round(4)
        )
        .drop("_records")
    )

    return (
        global_pct.collect()
        .join(slopes, on=["department", "province"], how="left")
        .sort("pct_silence", descending=True)
    )


# --------------------------------------------------------------------------- #
# Pipelines por tabla
# --------------------------------------------------------------------------- #
#: Cada entrada describe COMO deduplicar y CUALES columnas pasar through.
CLEANING_PLAN: dict[str, dict] = {
    "iras_weekly_dept": {
        "raw": "iras_weekly_dept.parquet",
        "key": ["department", "week_start"],
        "metrics": [
            "cases_total", "hosp_total", "deaths_total",
            "pneumonia_under5", "pneumonia_60plus",
            "hosp_under5", "hosp_60plus",
            "deaths_under5", "deaths_60plus",
        ],
        "passthrough": ["ubigeo", "ano", "semana"],
    },
    "iras_weekly_prov": {
        "raw": "iras_weekly_prov.parquet",
        "key": ["department", "province", "week_start"],
        "metrics": [
            "cases_total", "hosp_total", "deaths_total",
            "pneumonia_under5", "pneumonia_60plus",
            "hosp_under5", "hosp_60plus",
            "deaths_under5", "deaths_60plus",
        ],
        "passthrough": ["ubigeo", "ano", "semana"],
    },
    "pneumonia_dept": {
        "raw": "pneumonia_weekly_incidence_dept.parquet",
        "key": ["department", "week_start"],
        "metrics": [
            "cases_under5", "cases_60plus", "cases_total",
            "hosp_under5", "hosp_60plus", "hosp_total",
            "deaths_under5", "deaths_60plus", "deaths_total",
            "incidence_cases_under5", "incidence_cases_60plus", "incidence_cases_total",
            "incidence_hosp_under5", "incidence_hosp_60plus", "incidence_hosp_total",
            "incidence_deaths_under5", "incidence_deaths_60plus", "incidence_deaths_total",
        ],
        "passthrough": ["ubigeo", "ano", "semana"],
    },
    "pneumonia_prov": {
        "raw": "pneumonia_weekly_incidence_prov.parquet",
        "key": ["department", "province", "week_start"],
        "metrics": [
            "cases_under5", "cases_60plus", "cases_total",
            "hosp_under5", "hosp_60plus", "hosp_total",
            "deaths_under5", "deaths_60plus", "deaths_total",
            "incidence_cases_under5", "incidence_cases_60plus", "incidence_cases_total",
            "incidence_hosp_under5", "incidence_hosp_60plus", "incidence_hosp_total",
            "incidence_deaths_under5", "incidence_deaths_60plus", "incidence_deaths_total",
        ],
        "passthrough": ["ubigeo", "ano", "semana"],
    },
}


def clean_table(name: str, plan: dict) -> dict:
    """Aplica R1, R3 y R4 a una tabla y la guarda en data/processed/."""
    src = RAW_DIR / plan["raw"]
    if not src.exists():
        log(f"!! No existe {src}, se omite '{name}'.")
        return {}

    log(f"--- Limpiando '{name}' ({plan['raw']}) ---")
    lf = pl.scan_parquet(src)
    n_in = lf.select(pl.len()).collect().item()

    lf = deduplicate(lf, plan["key"], plan["metrics"], plan["passthrough"])
    lf = flag_death_gt_hosp(lf)
    lf = clamp_hosp_gt_cases(lf)

    df = lf.collect()
    n_out = df.height
    n_death_flag = int(df.get_column("flag_death_no_hosp").sum())
    n_hosp_fix = int(df.get_column("flag_hosp_corrected").sum())

    out_path = PROCESSED_DIR / f"{name}_cleaned.parquet"
    df.write_parquet(out_path)
    log(f"   filas {n_in:,} -> {n_out:,}   (-{n_in - n_out:,} duplicados)")
    log(f"   flag_death_no_hosp:  {n_death_flag:,}")
    log(f"   flag_hosp_corrected: {n_hosp_fix:,}")
    log(f"   guardado en: {out_path.relative_to(BASE_DIR)}")

    return {
        "name": name, "in": n_in, "out": n_out,
        "dups_removed": n_in - n_out,
        "flag_death_no_hosp": n_death_flag,
        "flag_hosp_corrected": n_hosp_fix,
        "path": out_path,
    }


# --------------------------------------------------------------------------- #
# Reporte ejecutivo de limpieza
# --------------------------------------------------------------------------- #
def write_cleaning_report(results: list[dict],
                          silence_df: pl.DataFrame | None) -> None:
    lines = ["=" * 84,
             "REPORTE DE LIMPIEZA (CLEANING) - INTEGRIDAD",
             "=" * 84, "",
             "Cambios aplicados a cada tabla raw:",
             "  R1. Deduplicacion por clave natural (max() en metricas).",
             "  R3. flag_death_no_hosp marcado (no eliminado).",
             "  R4. clamp cases_total = hosp_total cuando hosp_total > cases.",
             ""]
    for r in results:
        if not r:
            continue
        lines.append(f"## {r['name']}")
        lines.append(f"   filas:               {r['in']:>10,}  ->  {r['out']:>10,} "
                     f"(removidas: {r['dups_removed']:,})")
        lines.append(f"   flag_death_no_hosp:  {r['flag_death_no_hosp']:>10,}")
        lines.append(f"   flag_hosp_corrected: {r['flag_hosp_corrected']:>10,}")
        lines.append(f"   output:              {r['path'].relative_to(BASE_DIR)}")
        lines.append("")

    if silence_df is not None and silence_df.height:
        lines.append("=" * 84)
        lines.append("R2. SILENCIOS POR PROVINCIA (Top-15 con mayor sub-registro)")
        lines.append("=" * 84)
        lines.append(f"   {'department':<22s} {'province':<28s} "
                     f"{'pct_silence':>11s} {'slope/anio':>10s} {'n_obs':>8s}")
        for row in silence_df.head(15).iter_rows(named=True):
            lines.append(f"   {row['department'][:22]:<22s} "
                         f"{row['province'][:28]:<28s} "
                         f"{row['pct_silence']:>10.2f}% "
                         f"{row['slope_pct_per_year']:>+10.3f} "
                         f"{row['n_obs']:>8,}")
        lines.append("")
        lines.append("Lectura: slope_pct_per_year > 0 => el sub-registro EMPEORA;")
        lines.append("         slope_pct_per_year < 0 => MEJORA con los anios.")

    txt = "\n".join(lines)
    out = REPORTS_DIR / "cleaning_report.txt"
    out.write_text(txt, encoding="utf-8")
    print("\n" + txt)
    log(f"reporte cleaning -> {out.relative_to(BASE_DIR)}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    log(f"Polars {pl.__version__}  |  threads={pl.thread_pool_size()}")
    t0 = time.time()

    results = [clean_table(name, plan) for name, plan in CLEANING_PLAN.items()]

    # R2: solo aplica a la tabla con columna `province`
    prov_path = RAW_DIR / "iras_weekly_prov.parquet"
    silence_df = None
    if prov_path.exists():
        log("--- R2: Calculando silencios por provincia ---")
        silence_df = silence_by_province(prov_path)
        out = REPORTS_DIR / "silence_by_province.csv"
        silence_df.write_csv(out)
        log(f"   tabla -> {out.relative_to(BASE_DIR)}  ({silence_df.height} provincias)")

    write_cleaning_report(results, silence_df)
    log(f"Listo en {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
