"""
Validacion de Integridad de Datos - Hospitalizaciones y Decesos
================================================================

Proyecto : UNSA & Oklahoma - Team B (Salud y Sociedad)
Modulo   : src.integrity.validation
Ejecutar : python -X utf8 -m src.integrity.validation
Entradas : data/raw/*.parquet
Salidas  : reports/integrity/*

Objetivo
--------
Inspeccionar las 5 tablas con metricas de hospitalizacion (`hosp_*`) y
mortalidad (`deaths_*`) ANTES del modelado. Genera un reporte ejecutivo,
una matriz consolidada CSV y figuras (heatmaps + barras) para que el
equipo decida que filas corregir, marcar como flag o descartar.

Checks ejecutados
-----------------
+-----+--------------------------------+------------------------------------+
|  #  | Check                          | Significado                        |
+-----+--------------------------------+------------------------------------+
|  1  | `_chk_null__<col>`             | Valor null en hosp_*/deaths_*      |
|  2  | `_chk_neg__<col>`              | Valor negativo (imposible)         |
|  3  | `_chk_strata__hosp`            | under5 + 60plus > total            |
|  3  | `_chk_strata__deaths`          | under5 + 60plus > total            |
|  4  | `_chk_logic__hosp_gt_cases`    | hospitalizados > casos reportados  |
|  4  | `_chk_logic__deaths_gt_cases`  | muertes > casos reportados         |
|  4  | `_chk_logic__deaths_gt_hosp`   | muertes > hospitalizaciones        |
|  5  | duplicados                     | (ubigeo, week_start) repetida      |
|  6  | `_chk_range__year`             | ano fuera de [1999, 2026]          |
|  7  | `_chk_range__week`             | semana fuera de [1, 53]            |
|  8  | gaps                           | grupos con cobertura < 95% semanas |
|  9  | `_chk_silence__...`            | casos>0 sin hosp y sin deaths      |
+-----+--------------------------------+------------------------------------+

Salidas
-------
- ``integrity_report.txt``        : Resumen legible.
- ``integrity_overview.csv``      : Matriz tabla x check (todos los conteos
                                    y % en una sola fila por tabla).
- ``integrity_summary_<t>.csv``   : Nulls por columna (no solo hosp/death).
- ``integrity_gaps_<t>.csv``      : Cobertura temporal por grupo.
- ``integrity_issues_<t>.parquet``: Filas con >=1 check positivo.
- ``figures/integrity_nulls_<t>.png``   : Heatmap dept x columna.
- ``figures/integrity_timeline_<t>.png``: Evolucion % nulos por anio.
- ``figures/integrity_issues_<t>.png``  : Barras (log) con cada hallazgo.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

# --------------------------------------------------------------------------- #
# Configuracion
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
REPORTS_DIR = BASE_DIR / "reports" / "integrity"
FIG_DIR = REPORTS_DIR / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

#: Cada tabla declara sus columnas clave para los checks.
TARGETS: dict[str, dict] = {
    "iras_raw": {
        "path": RAW_DIR / "iras_data_raw.parquet",
        "group_keys": ["department"],
        "key_cols": ["ubigeo", "ano", "semana"],
        "hosp_cols": ["hosp_under5", "hosp_60plus", "hosp_total"],
        "death_cols": ["deaths_under5", "deaths_60plus", "deaths_total"],
        "cases_col": "cases_total",
        "year_col": "ano", "week_col": "semana", "date_col": "week_start",
    },
    "iras_weekly_dept": {
        "path": RAW_DIR / "iras_weekly_dept.parquet",
        "group_keys": ["department"], "key_cols": ["ubigeo", "week_start"],
        "hosp_cols": ["hosp_under5", "hosp_60plus", "hosp_total"],
        "death_cols": ["deaths_under5", "deaths_60plus", "deaths_total"],
        "cases_col": "cases_total",
        "year_col": "ano", "week_col": "semana", "date_col": "week_start",
    },
    "iras_weekly_prov": {
        "path": RAW_DIR / "iras_weekly_prov.parquet",
        "group_keys": ["department"], "key_cols": ["ubigeo", "week_start"],
        "hosp_cols": ["hosp_under5", "hosp_60plus", "hosp_total"],
        "death_cols": ["deaths_under5", "deaths_60plus", "deaths_total"],
        "cases_col": "cases_total",
        "year_col": "ano", "week_col": "semana", "date_col": "week_start",
    },
    "pneumonia_dept": {
        "path": RAW_DIR / "pneumonia_weekly_incidence_dept.parquet",
        "group_keys": ["department"], "key_cols": ["ubigeo", "week_start"],
        "hosp_cols": ["hosp_under5", "hosp_60plus", "hosp_total"],
        "death_cols": ["deaths_under5", "deaths_60plus", "deaths_total"],
        "cases_col": "cases_total",
        "year_col": "ano", "week_col": "semana", "date_col": "week_start",
    },
    "pneumonia_prov": {
        "path": RAW_DIR / "pneumonia_weekly_incidence_prov.parquet",
        "group_keys": ["department"], "key_cols": ["ubigeo", "week_start"],
        "hosp_cols": ["hosp_under5", "hosp_60plus", "hosp_total"],
        "death_cols": ["deaths_under5", "deaths_60plus", "deaths_total"],
        "cases_col": "cases_total",
        "year_col": "ano", "week_col": "semana", "date_col": "week_start",
    },
}

YEAR_MIN, YEAR_MAX = 1999, 2026
WEEK_MIN, WEEK_MAX = 1, 53


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Construccion de banderas (una columna `_chk_*` por regla)
# --------------------------------------------------------------------------- #
def add_flags(lf: pl.LazyFrame, cfg: dict) -> tuple[pl.LazyFrame, list[str]]:
    """Agrega columnas booleanas `_chk_*` que marcan filas problematicas
    bajo las 9 reglas descritas en el docstring del modulo.

    El uso de `fill_null(0)` antes de las comparaciones evita falsos
    positivos en `_chk_neg__*` cuando la columna ya tiene nulls (que ya
    fueron contados por `_chk_null__*`).
    """
    hosp, deaths = cfg["hosp_cols"], cfg["death_cols"]
    cases = cfg["cases_col"]
    yc, wc = cfg["year_col"], cfg["week_col"]

    flags: list[str] = []
    expr: list[pl.Expr] = []

    # 1. NULOS en hosp/deaths
    for c in hosp + deaths:
        name = f"_chk_null__{c}"
        expr.append(pl.col(c).is_null().alias(name)); flags.append(name)

    # 2. NEGATIVOS en hosp/deaths
    for c in hosp + deaths:
        name = f"_chk_neg__{c}"
        expr.append((pl.col(c).fill_null(0) < 0).alias(name)); flags.append(name)

    # 3. SUMA ESTRATOS > TOTAL (under5 + 60plus > total => inconsistente)
    expr.append(
        ((pl.col("hosp_under5").fill_null(0) + pl.col("hosp_60plus").fill_null(0))
         > pl.col("hosp_total").fill_null(0)).alias("_chk_strata__hosp"))
    flags.append("_chk_strata__hosp")
    expr.append(
        ((pl.col("deaths_under5").fill_null(0) + pl.col("deaths_60plus").fill_null(0))
         > pl.col("deaths_total").fill_null(0)).alias("_chk_strata__deaths"))
    flags.append("_chk_strata__deaths")

    # 4. COHERENCIA LOGICA
    expr.append((pl.col("hosp_total").fill_null(0) > pl.col(cases).fill_null(0))
                .alias("_chk_logic__hosp_gt_cases"))
    flags.append("_chk_logic__hosp_gt_cases")
    expr.append((pl.col("deaths_total").fill_null(0) > pl.col(cases).fill_null(0))
                .alias("_chk_logic__deaths_gt_cases"))
    flags.append("_chk_logic__deaths_gt_cases")
    expr.append((pl.col("deaths_total").fill_null(0) > pl.col("hosp_total").fill_null(0))
                .alias("_chk_logic__deaths_gt_hosp"))
    flags.append("_chk_logic__deaths_gt_hosp")

    # 6. RANGOS DE FECHA
    expr.append(((pl.col(yc) < YEAR_MIN) | (pl.col(yc) > YEAR_MAX))
                .alias("_chk_range__year"))
    flags.append("_chk_range__year")
    expr.append(((pl.col(wc) < WEEK_MIN) | (pl.col(wc) > WEEK_MAX))
                .alias("_chk_range__week"))
    flags.append("_chk_range__week")

    # 9. SILENCIO SOSPECHOSO: casos>0 sin hosp y sin deaths
    expr.append(
        ((pl.col(cases).fill_null(0) > 0)
         & (pl.col("hosp_total").fill_null(0) == 0)
         & (pl.col("deaths_total").fill_null(0) == 0))
        .alias("_chk_silence__no_hosp_no_death_with_cases"))
    flags.append("_chk_silence__no_hosp_no_death_with_cases")

    return lf.with_columns(expr), flags


def check_duplicates(lf: pl.LazyFrame, key_cols: list[str]) -> int:
    """Cuenta filas REDUNDANTES (en grupos de tamano > 1 contamos n-1)."""
    grp = lf.group_by(key_cols).agg(pl.len().alias("_n")).filter(pl.col("_n") > 1)
    total = grp.select(pl.col("_n").sum()).collect().item() or 0
    n_groups = grp.select(pl.len()).collect().item() or 0
    return int(total) - int(n_groups)


def check_temporal_gaps(lf: pl.LazyFrame, group_keys: list[str],
                        date_col: str) -> pl.DataFrame:
    """Cobertura temporal por grupo: cuenta semanas observadas vs
    esperadas (floor((max - min)/7) + 1) y devuelve % de cobertura."""
    df = (
        lf.select(group_keys + [date_col]).unique()
        .group_by(group_keys)
        .agg(pl.col(date_col).min().alias("date_min"),
             pl.col(date_col).max().alias("date_max"),
             pl.col(date_col).n_unique().alias("n_observado"))
        .collect()
    )
    return df.with_columns(
        n_esperado=((pl.col("date_max") - pl.col("date_min")).dt.total_days() // 7 + 1)
    ).with_columns(
        pct_cobertura=(pl.col("n_observado") / pl.col("n_esperado") * 100).round(2),
        n_faltantes=(pl.col("n_esperado") - pl.col("n_observado")),
    ).sort("pct_cobertura")


def null_heatmap_data(df: pl.DataFrame, cols: list[str],
                      group_keys: list[str]) -> pl.DataFrame:
    return (df.group_by(group_keys, maintain_order=True)
            .agg([(pl.col(c).is_null().sum() / pl.len() * 100).round(2).alias(c)
                  for c in cols])
            .sort(group_keys))


def null_timeline(df: pl.DataFrame, cols: list[str], year_col: str) -> pl.DataFrame:
    return (df.group_by(year_col, maintain_order=True)
            .agg([(pl.col(c).is_null().sum() / pl.len() * 100).round(2).alias(c)
                  for c in cols])
            .sort(year_col))


# --------------------------------------------------------------------------- #
# Visualizacion
# --------------------------------------------------------------------------- #
def plot_null_heatmap(data: pl.DataFrame, cols: list[str],
                      group_keys: list[str], title: str, out: Path) -> None:
    if data.height == 0:
        return
    labels = data.get_column(group_keys[0]).to_list()
    mat = data.select(cols).to_numpy().astype(float)
    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 1.1),
                                    max(5, len(labels) * 0.32)))
    im = ax.imshow(mat, aspect="auto", cmap="Reds",
                   vmin=0, vmax=max(mat.max(), 1))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if v > 0:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if v > mat.max() / 2 else "black")
    plt.colorbar(im, ax=ax, shrink=0.7).set_label("% nulos", fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    log(f"   figura -> {out.relative_to(BASE_DIR)}")


def plot_issue_bars(totals: dict, n_rows: int, dup_rows: int,
                    title: str, out: Path) -> None:
    """Barras horizontales en escala log con los checks que SI activaron."""
    items = [(k, v) for k, v in totals.items() if v > 0]
    if dup_rows:
        items.append(("_chk_duplicates", dup_rows))
    if not items:
        return
    items.sort(key=lambda kv: kv[1])
    labels = [k.replace("_chk_", "") for k, _ in items]
    counts = [v for _, v in items]
    pcts = [100 * v / n_rows for v in counts]
    fig, ax = plt.subplots(figsize=(11, max(3, len(items) * 0.45)))
    bars = ax.barh(labels, counts, color="#d62728", alpha=0.78,
                   edgecolor="black", linewidth=0.4)
    ax.set_xscale("log")
    ax.set_xlabel("Filas afectadas (escala log)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, which="both")
    for bar, c, p in zip(bars, counts, pcts):
        ax.text(bar.get_width() * 1.05, bar.get_y() + bar.get_height() / 2,
                f"{c:,}  ({p:.2f}%)", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    log(f"   figura -> {out.relative_to(BASE_DIR)}")


def plot_null_timeline(data: pl.DataFrame, cols: list[str], year_col: str,
                       title: str, out: Path) -> None:
    if data.height == 0:
        return
    years = data.get_column(year_col).to_list()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for c in cols:
        ax.plot(years, data.get_column(c).to_list(),
                marker="o", lw=1.4, ms=4, label=c)
    ax.set_xlabel("Anio"); ax.set_ylabel("% nulos")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    log(f"   figura -> {out.relative_to(BASE_DIR)}")


# --------------------------------------------------------------------------- #
# Pipeline por tabla
# --------------------------------------------------------------------------- #
def analyze(name: str, cfg: dict) -> dict:
    """Ejecuta TODOS los checks sobre una tabla y persiste resultados."""
    log(f"--- Validando '{name}' ({cfg['path'].name}) ---")
    lf = pl.scan_parquet(cfg["path"])

    n_rows = lf.select(pl.len()).collect().item()
    log(f"   filas: {n_rows:,}")

    lf, flag_cols = add_flags(lf, cfg)
    totals = (
        lf.select([pl.col(c).sum().cast(pl.Int64).alias(c) for c in flag_cols])
        .collect().row(0, named=True)
    )

    dup_rows = check_duplicates(pl.scan_parquet(cfg["path"]), cfg["key_cols"])
    gaps = check_temporal_gaps(pl.scan_parquet(cfg["path"]),
                               cfg["group_keys"], cfg["date_col"])
    gaps.write_csv(REPORTS_DIR / f"integrity_gaps_{name}.csv")

    # Nulos por columna (todas las columnas, no solo hosp/death)
    all_cols = pl.scan_parquet(cfg["path"]).collect_schema().names()
    null_summary = (
        pl.scan_parquet(cfg["path"])
        .select([pl.col(c).null_count().cast(pl.Int64).alias(c)
                 for c in all_cols])
        .collect()
        .transpose(include_header=True, header_name="column",
                   column_names=["nulls"])
        .with_columns(pct=(pl.col("nulls") / n_rows * 100).round(3))
        .sort("nulls", descending=True)
    )
    null_summary.write_csv(REPORTS_DIR / f"integrity_summary_{name}.csv")

    any_flag = pl.fold(acc=pl.lit(False), function=lambda a, b: a | b,
                       exprs=[pl.col(c) for c in flag_cols])
    detail = (lf.filter(any_flag)
              .select(cfg["group_keys"] + cfg["key_cols"]
                      + cfg["hosp_cols"] + cfg["death_cols"]
                      + [cfg["cases_col"]] + flag_cols)
              .collect())
    detail.write_parquet(REPORTS_DIR / f"integrity_issues_{name}.parquet")

    # Figuras
    df_full = pl.read_parquet(cfg["path"])
    cols_for_heatmap = cfg["hosp_cols"] + cfg["death_cols"]
    plot_null_heatmap(
        null_heatmap_data(df_full, cols_for_heatmap, cfg["group_keys"]),
        cols_for_heatmap, cfg["group_keys"],
        f"% Nulos por {cfg['group_keys'][0]} - {name}",
        FIG_DIR / f"integrity_nulls_{name}.png")
    plot_null_timeline(
        null_timeline(df_full, cols_for_heatmap, cfg["year_col"]),
        cols_for_heatmap, cfg["year_col"],
        f"Evolucion del % de nulos por anio - {name}",
        FIG_DIR / f"integrity_timeline_{name}.png")
    plot_issue_bars(totals, n_rows, dup_rows,
                    f"Issues de integridad detectados - {name}",
                    FIG_DIR / f"integrity_issues_{name}.png")

    return {"name": name, "n_rows": n_rows, "totals": totals,
            "dup_rows": dup_rows, "gaps": gaps,
            "null_summary": null_summary, "issues": detail.height,
            "cfg": cfg}


# --------------------------------------------------------------------------- #
# Reporte consolidado
# --------------------------------------------------------------------------- #
def build_overview(results: list[dict]) -> pl.DataFrame:
    rows = []
    for r in results:
        n = r["n_rows"]
        row = {"tabla": r["name"], "filas": n, "duplicados": r["dup_rows"]}
        for k, v in r["totals"].items():
            row[k] = v
            row[f"{k}_pct"] = round(100 * v / n, 3) if n else 0.0
        rows.append(row)
    return pl.DataFrame(rows)


def executive_report(results: list[dict]) -> None:
    lines = ["=" * 84,
             "REPORTE DE INTEGRIDAD - HOSPITALIZACIONES Y DECESOS",
             "=" * 84, ""]
    for r in results:
        n = r["n_rows"]
        lines.append(f"## Tabla: {r['name']}    filas={n:,}    "
                     f"issues={r['issues']:,}    duplicados={r['dup_rows']:,}")
        lines.append("   ----- Checks (conteo / %) -----")
        for k, v in sorted(r["totals"].items(), key=lambda kv: -kv[1]):
            pct = 100 * v / n if n else 0
            mark = "!!" if pct >= 1.0 else "  "
            lines.append(f"   {mark} {k:<55s} {v:>10,}  ({pct:6.3f}%)")
        top_nulls = r["null_summary"].filter(pl.col("nulls") > 0).head(5)
        if top_nulls.height:
            lines.append("   ----- Top 5 columnas con mas nulos -----")
            for row in top_nulls.iter_rows(named=True):
                lines.append(f"      - {row['column']:<35s} "
                             f"{row['nulls']:>10,}  ({row['pct']:6.3f}%)")
        gaps = r["gaps"].filter(pl.col("pct_cobertura") < 95)
        if gaps.height:
            lines.append("   ----- Grupos con cobertura temporal < 95% -----")
            for row in gaps.iter_rows(named=True):
                grp = " / ".join(str(row[k]) for k in r["cfg"]["group_keys"])
                lines.append(f"      - {grp:<35s} "
                             f"{row['pct_cobertura']:>6.2f}%  "
                             f"(faltan {row['n_faltantes']} semanas)")
        lines.append("")
    txt = "\n".join(lines)
    (REPORTS_DIR / "integrity_report.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt)
    log(f"reporte ejecutivo -> reports/integrity/integrity_report.txt")


def main() -> None:
    log(f"Polars {pl.__version__}  |  threads={pl.thread_pool_size()}")
    t0 = time.time()
    results = []
    for name, cfg in TARGETS.items():
        if not cfg["path"].exists():
            log(f"!! No existe {cfg['path']}, se omite.")
            continue
        results.append(analyze(name, cfg))
    overview = build_overview(results)
    overview.write_csv(REPORTS_DIR / "integrity_overview.csv")
    log(f"overview -> reports/integrity/integrity_overview.csv")
    executive_report(results)
    log(f"Listo en {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
