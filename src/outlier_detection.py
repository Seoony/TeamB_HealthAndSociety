"""
Deteccion de Valores Atipicos (Outliers) y Anomalias
====================================================

Proyecto: UNSA & Oklahoma - Team B (Salud y Sociedad)
Dataset:  IRAs y Neumonia - series semanales por departamento/provincia.

Metodos implementados (todos vectorizados sobre Polars LazyFrame):
  1. IQR (Tukey)            - por grupo (departamento)
  2. Modified Z-score (MAD) - robusto a colas pesadas
  3. Rolling Z-score        - desviaciones locales en la serie temporal
  4. Baseline estacional    - residuo vs mediana semana-del-anio + MAD
  5. Isolation Forest       - deteccion multivariada (sklearn)

Salidas:
  - reports/outliers_summary_<tabla>.csv
  - reports/outliers_detail_<tabla>.parquet
  - reports/figures/outliers_<dept>_<metrica>.png   (con etiqueta numerica
    junto a cada outlier marcado para verlo de un vistazo)
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from sklearn.ensemble import IsolationForest

# --------------------------------------------------------------------------- #
# Configuracion
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
REPORTS_DIR = BASE_DIR / "reports"
FIG_DIR = REPORTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Tablas objetivo (series semanales)
TARGETS = {
    "pneumonia_dept": {
        "path": RAW_DIR / "pneumonia_weekly_incidence_dept.parquet",
        "group_keys": ["department"],
        "metrics": [
            "cases_total",
            "hosp_total",
            "deaths_total",
            "incidence_cases_total",
            "incidence_hosp_total",
        ],
    },
    "iras_dept": {
        "path": RAW_DIR / "iras_weekly_dept.parquet",
        "group_keys": ["department"],
        "metrics": ["cases_total", "hosp_total", "deaths_total"],
    },
}

# Umbrales por metodo
IQR_K = 1.5             # Tukey clasico (3.0 = extremo)
MAD_THRESHOLD = 3.5     # Iglewicz & Hoaglin (1993)
ROLLING_WINDOW = 13     # ~3 meses (semanas)
ROLLING_THRESHOLD = 3.0
SEASONAL_THRESHOLD = 3.5
IFOREST_CONTAMINATION = 0.01  # 1% esperado

# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    return pl.when(den == 0).then(None).otherwise(num / den)


# --------------------------------------------------------------------------- #
# Metodos univariados (Polars puro, perezoso)
# --------------------------------------------------------------------------- #
def flag_iqr(lf: pl.LazyFrame, group_keys: list[str], metric: str, k: float = IQR_K) -> pl.LazyFrame:
    """Devuelve LazyFrame con columnas <metric>_iqr_low/high/outlier."""
    q1 = pl.col(metric).quantile(0.25).over(group_keys)
    q3 = pl.col(metric).quantile(0.75).over(group_keys)
    iqr = q3 - q1
    low = q1 - k * iqr
    high = q3 + k * iqr
    return lf.with_columns(
        low.alias(f"{metric}__iqr_low"),
        high.alias(f"{metric}__iqr_high"),
        ((pl.col(metric) < low) | (pl.col(metric) > high)).alias(f"{metric}__is_iqr"),
    )


def flag_mad(lf: pl.LazyFrame, group_keys: list[str], metric: str,
             thr: float = MAD_THRESHOLD) -> pl.LazyFrame:
    """Modified Z-score = 0.6745 * (x - median) / MAD."""
    med = pl.col(metric).median().over(group_keys)
    mad = (pl.col(metric) - med).abs().median().over(group_keys)
    score = _safe_div(0.6745 * (pl.col(metric) - med), mad)
    return lf.with_columns(
        score.alias(f"{metric}__mad_score"),
        (score.abs() > thr).alias(f"{metric}__is_mad"),
    )


def flag_rolling_z(lf: pl.LazyFrame, group_keys: list[str], metric: str,
                   window: int = ROLLING_WINDOW, thr: float = ROLLING_THRESHOLD) -> pl.LazyFrame:
    """Z-score con media/desv movil (centrada)."""
    mean = (
        pl.col(metric)
        .rolling_mean(window_size=window, center=True, min_samples=max(3, window // 2))
        .over(group_keys)
    )
    std = (
        pl.col(metric)
        .rolling_std(window_size=window, center=True, min_samples=max(3, window // 2))
        .over(group_keys)
    )
    score = _safe_div(pl.col(metric) - mean, std)
    return lf.with_columns(
        score.alias(f"{metric}__roll_z"),
        (score.abs() > thr).alias(f"{metric}__is_roll"),
    )


def flag_seasonal(lf: pl.LazyFrame, group_keys: list[str], metric: str,
                  thr: float = SEASONAL_THRESHOLD) -> pl.LazyFrame:
    """Compara el valor con la mediana/MAD de su misma semana-del-anio."""
    keys = group_keys + ["semana"]
    med = pl.col(metric).median().over(keys)
    mad = (pl.col(metric) - med).abs().median().over(keys)
    score = _safe_div(0.6745 * (pl.col(metric) - med), mad)
    return lf.with_columns(
        score.alias(f"{metric}__seas_score"),
        (score.abs() > thr).alias(f"{metric}__is_seas"),
    )


# --------------------------------------------------------------------------- #
# Metodo multivariado: Isolation Forest
# --------------------------------------------------------------------------- #
def flag_isolation_forest(df: pl.DataFrame, group_keys: list[str], metrics: list[str],
                          contamination: float = IFOREST_CONTAMINATION) -> pl.DataFrame:
    """Aplica IsolationForest por grupo. Devuelve flag bool y score."""
    flags = []
    scores = []
    groups = df.group_by(group_keys, maintain_order=True)
    for _, sub in groups:
        X = sub.select(metrics).fill_null(0.0).to_numpy()
        if X.shape[0] < 30:  # poca data para entrenar
            flags.extend([False] * X.shape[0])
            scores.extend([0.0] * X.shape[0])
            continue
        iso = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        iso.fit(X)
        pred = iso.predict(X)            # -1 = anomalia
        sc = -iso.score_samples(X)       # mayor = mas anomalo
        flags.extend((pred == -1).tolist())
        scores.extend(sc.tolist())
    return df.with_columns(
        pl.Series("iforest_score", scores),
        pl.Series("is_iforest", flags),
    )


# --------------------------------------------------------------------------- #
# Pipeline por tabla
# --------------------------------------------------------------------------- #
def analyze(name: str, cfg: dict) -> dict:
    log(f"--- Analizando '{name}' ({cfg['path'].name}) ---")
    lf = pl.scan_parquet(cfg["path"])

    # Deduplicar por (grupo, week_start): la tabla raw replica el mismo total
    # 12+ veces (un registro por unidad ubigeo). Agregamos con max() para
    # quedarnos con un solo punto por semana sin perder magnitud.
    agg_cols = cfg["metrics"] + ["ano", "semana"]
    lf = (
        lf.group_by(cfg["group_keys"] + ["week_start"], maintain_order=False)
        .agg([pl.col(c).max() for c in agg_cols])
        .sort(cfg["group_keys"] + ["week_start"])
    )

    # Aplicar metodos univariados a cada metrica
    for metric in cfg["metrics"]:
        lf = flag_iqr(lf, cfg["group_keys"], metric)
        lf = flag_mad(lf, cfg["group_keys"], metric)
        lf = flag_rolling_z(lf, cfg["group_keys"], metric)
        lf = flag_seasonal(lf, cfg["group_keys"], metric)

    df = lf.collect()
    log(f"   filas analizadas: {df.height:,}  |  columnas: {df.width}")

    # Isolation Forest sobre las metricas como vector multivariado
    df = flag_isolation_forest(df, cfg["group_keys"], cfg["metrics"])

    # Resumen: conteo de flags por metodo
    flag_cols = [c for c in df.columns if c.startswith(tuple(cfg["metrics"]))
                 and ("__is_" in c)]
    flag_cols.append("is_iforest")

    summary = (
        df.group_by(cfg["group_keys"], maintain_order=True)
        .agg([pl.col(c).sum().cast(pl.Int64).alias(c) for c in flag_cols]
             + [pl.len().alias("n_obs")])
    )
    summary_path = REPORTS_DIR / f"outliers_summary_{name}.csv"
    summary.write_csv(summary_path)
    log(f"   resumen      -> {summary_path.relative_to(BASE_DIR)}")

    # Detalle: solo filas con AL MENOS un metodo activado
    any_flag = pl.fold(
        acc=pl.lit(False),
        function=lambda a, b: a | b,
        exprs=[pl.col(c) for c in flag_cols],
    )
    detail = df.filter(any_flag).select(
        cfg["group_keys"] + ["ano", "semana", "week_start"] + cfg["metrics"] + flag_cols
        + ["iforest_score"]
    )
    detail_path = REPORTS_DIR / f"outliers_detail_{name}.parquet"
    detail.write_parquet(detail_path)
    log(f"   detalle      -> {detail_path.relative_to(BASE_DIR)} ({detail.height:,} filas)")

    return {"name": name, "df": df, "summary": summary, "flag_cols": flag_cols, "cfg": cfg}


# --------------------------------------------------------------------------- #
# Visualizacion
# --------------------------------------------------------------------------- #
def plot_top(result: dict, metric: str, top_n: int = 4,
             label_top_k: int = 8) -> None:
    """Genera un PNG por cada departamento top. Junto a cada outlier coloca
    el VALOR NUMERICO de la metrica, para que al ver la imagen sepas cuanto
    fue exactamente. Se etiquetan los `label_top_k` valores mas altos por
    metodo para no saturar la grafica."""
    name, df, cfg = result["name"], result["df"], result["cfg"]
    group_keys = cfg["group_keys"]

    flag_col = f"{metric}__is_mad"
    if flag_col not in df.columns:
        return
    top = (
        df.group_by(group_keys)
        .agg(pl.col(flag_col).sum().alias("n"))
        .sort("n", descending=True)
        .head(top_n)
        .get_column(group_keys[0])
        .to_list()
    )

    colors = {"IQR": "#ff7f0e", "MAD": "#2ca02c", "Rolling": "#d62728",
              "Seasonal": "#9467bd", "IForest": "#8c564b"}

    for dept in top:
        sub = df.filter(pl.col(group_keys[0]) == dept).sort("week_start")
        x = sub.get_column("week_start").to_list()
        y = sub.get_column(metric).to_list()

        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(x, y, color="#1f77b4", lw=0.9, label=metric, alpha=0.75)

        masks = {
            "IQR":      sub.get_column(f"{metric}__is_iqr").to_list(),
            "MAD":      sub.get_column(f"{metric}__is_mad").to_list(),
            "Rolling":  sub.get_column(f"{metric}__is_roll").to_list(),
            "Seasonal": sub.get_column(f"{metric}__is_seas").to_list(),
            "IForest":  sub.get_column("is_iforest").to_list(),
        }

        # Para no etiquetar todo, guardamos los top_k mas altos de cada metodo
        labels_to_show: list[tuple] = []  # (x, y, color)
        for label, mask in masks.items():
            xs = [xi for xi, m in zip(x, mask) if m]
            ys = [yi for yi, m in zip(y, mask) if m]
            if not xs:
                continue
            ax.scatter(xs, ys, s=28, color=colors[label],
                       label=f"{label} ({len(xs)})",
                       alpha=0.85, edgecolor="black", linewidth=0.4, zorder=3)

            # Top-k por valor (los picos mas notorios)
            order = sorted(range(len(ys)), key=lambda i: ys[i], reverse=True)[:label_top_k]
            for i in order:
                labels_to_show.append((xs[i], ys[i], colors[label]))

        # Dibujamos los numeros junto a cada outlier destacado
        for xi, yi, color in labels_to_show:
            ax.annotate(
                f"{yi:,.0f}",
                xy=(xi, yi),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=7,
                color=color,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec=color, lw=0.5, alpha=0.85),
            )

        ax.set_title(f"Outliers - {dept} - {metric}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Semana")
        ax.set_ylabel(metric)
        ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
        ax.grid(alpha=0.3)
        fig.tight_layout()

        safe_dept = "".join(c if c.isalnum() else "_" for c in dept)[:40]
        out = FIG_DIR / f"outliers_{name}_{safe_dept}_{metric}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        log(f"   figura       -> {out.relative_to(BASE_DIR)}")


# --------------------------------------------------------------------------- #
# Reporte ejecutivo
# --------------------------------------------------------------------------- #
def executive_report(results: list[dict]) -> None:
    lines = ["=" * 78, "REPORTE EJECUTIVO - DETECCION DE OUTLIERS Y ANOMALIAS", "=" * 78, ""]
    for r in results:
        name, df, summary, flag_cols = r["name"], r["df"], r["summary"], r["flag_cols"]
        lines.append(f"## Tabla: {name}    filas={df.height:,}")
        totals = {c: int(df.get_column(c).sum()) for c in flag_cols}
        for c, v in sorted(totals.items(), key=lambda kv: -kv[1]):
            pct = 100 * v / df.height
            lines.append(f"   {c:<55s} {v:>8,}  ({pct:5.2f}%)")
        # top 5 grupos con mas anomalias multivariadas
        top = (
            summary.sort("is_iforest", descending=True)
            .head(5)
            .select(r["cfg"]["group_keys"] + ["is_iforest", "n_obs"])
        )
        lines.append("   Top-5 grupos por IForest:")
        for row in top.iter_rows(named=True):
            grp = " / ".join(str(row[k]) for k in r["cfg"]["group_keys"])
            lines.append(f"      - {grp:<35s} {row['is_iforest']:>5} / {row['n_obs']:>5}")
        lines.append("")
    txt = "\n".join(lines)
    out = REPORTS_DIR / "outliers_report.txt"
    out.write_text(txt, encoding="utf-8")
    print("\n" + txt)
    log(f"reporte ejecutivo -> {out.relative_to(BASE_DIR)}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    log(f"Polars {pl.__version__}  |  threads={pl.thread_pool_size()}")
    t0 = time.time()

    results = []
    for name, cfg in TARGETS.items():
        if not cfg["path"].exists():
            log(f"!! No existe {cfg['path']}, se omite.")
            continue
        results.append(analyze(name, cfg))

    # Figuras para la metrica clave
    for r in results:
        primary = "cases_total" if "cases_total" in r["cfg"]["metrics"] else r["cfg"]["metrics"][0]
        plot_top(r, primary, top_n=4)

    executive_report(results)
    log(f"Listo en {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
