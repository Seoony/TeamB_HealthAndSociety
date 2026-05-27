"""
Deteccion de Valores Atipicos (Outliers) y Anomalias
=====================================================

Proyecto : UNSA & Oklahoma - Team B (Salud y Sociedad)
Modulo   : src.outliers.detection
Ejecutar : python -X utf8 -m src.outliers.detection
Entradas : data/raw/*.parquet
Salidas  : reports/outliers/*

Contexto
--------
Las series semanales de IRAs y Neumonia por departamento son la base del
modelado predictivo (ARIMA/SARIMA/XGBoost/Prophet/LSTM). Antes de modelar
necesitamos identificar *picos* que sean genuinos brotes y diferenciarlos
de ruido o errores de captura. Este script automatiza esa deteccion con
cinco metodos complementarios:

==========  =====================================================  ========
Metodo      Descripcion                                            Umbral
==========  =====================================================  ========
IQR         Tukey (Q1-1.5 IQR, Q3+1.5 IQR) por grupo.              k=1.5
MAD         Z-score modificado con mediana absoluta (Iglewicz).    >3.5
Rolling     Z-score con media/desv movil centrada de 13 semanas.   >3.0
Seasonal    Compara con la mediana/MAD de la misma semana-de-anio. >3.5
IForest     Isolation Forest multivariado (sklearn) por grupo.     1% cont.
==========  =====================================================  ========

Por que estos cinco?
- **IQR**: barato, intuitivo, equivalente a `boxplot`. Pero falla con
  distribuciones asimetricas (caso de IRAs estacionales).
- **MAD**: robusto a colas pesadas. Lo elegimos para complementar IQR.
- **Rolling Z**: captura anomalias *locales* (un pico raro en abril,
  cuando el resto de abriles fueron normales).
- **Seasonal**: captura anomalias *estacionales* (una semana 30 con
  valores fuera de lo esperado para esa misma semana en anios previos).
- **IForest**: multivariado. Detecta combinaciones extranas (p. ej.
  altas hospitalizaciones con pocos casos) que los univariados no ven.

Performance
-----------
Polars LazyFrame + `over()` ejecuta los 4 metodos univariados en una
sola pasada. IsolationForest es por grupo (sklearn). 28k filas se
procesan en ~25 s con `pl.thread_pool_size() = 8`.
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
BASE_DIR = Path(__file__).resolve().parents[2]   # .../TeamB_HealthAndSociety
RAW_DIR = BASE_DIR / "data" / "raw"
REPORTS_DIR = BASE_DIR / "reports" / "outliers"
FIG_DIR = REPORTS_DIR / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

#: Tablas objetivo. Cada entrada define la ruta, las columnas-clave de
#: agrupamiento y las metricas a las que se aplicaran los 5 metodos.
TARGETS: dict[str, dict] = {
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

# Umbrales (literatura + tuning empirico)
IQR_K = 1.5                  #: Multiplicador Tukey clasico (3.0 = extremo).
MAD_THRESHOLD = 3.5          #: Iglewicz & Hoaglin (1993).
ROLLING_WINDOW = 13          #: ~3 meses de ventana (semanas).
ROLLING_THRESHOLD = 3.0      #: |z| > 3 en serie movil.
SEASONAL_THRESHOLD = 3.5     #: |z| > 3.5 vs misma semana de anio.
IFOREST_CONTAMINATION = 0.01 #: 1% de filas etiquetadas como anomalia.


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    """Logger minimalista con timestamp para ejecuciones largas."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    """Division segura: si el denominador es 0, devuelve null (no error)."""
    return pl.when(den == 0).then(None).otherwise(num / den)


# --------------------------------------------------------------------------- #
# Metodos univariados
# --------------------------------------------------------------------------- #
def flag_iqr(lf: pl.LazyFrame, group_keys: list[str], metric: str,
             k: float = IQR_K) -> pl.LazyFrame:
    """Tukey IQR por grupo.

    Marca como outlier todo valor fuera de [Q1 - k*IQR, Q3 + k*IQR]
    calculado dentro de cada grupo (departamento).
    """
    q1 = pl.col(metric).quantile(0.25).over(group_keys)
    q3 = pl.col(metric).quantile(0.75).over(group_keys)
    iqr = q3 - q1
    low, high = q1 - k * iqr, q3 + k * iqr
    return lf.with_columns(
        low.alias(f"{metric}__iqr_low"),
        high.alias(f"{metric}__iqr_high"),
        ((pl.col(metric) < low) | (pl.col(metric) > high)).alias(f"{metric}__is_iqr"),
    )


def flag_mad(lf: pl.LazyFrame, group_keys: list[str], metric: str,
             thr: float = MAD_THRESHOLD) -> pl.LazyFrame:
    """Modified Z-score por grupo: z = 0.6745 * (x - mediana) / MAD."""
    med = pl.col(metric).median().over(group_keys)
    mad = (pl.col(metric) - med).abs().median().over(group_keys)
    score = _safe_div(0.6745 * (pl.col(metric) - med), mad)
    return lf.with_columns(
        score.alias(f"{metric}__mad_score"),
        (score.abs() > thr).alias(f"{metric}__is_mad"),
    )


def flag_rolling_z(lf: pl.LazyFrame, group_keys: list[str], metric: str,
                   window: int = ROLLING_WINDOW,
                   thr: float = ROLLING_THRESHOLD) -> pl.LazyFrame:
    """Z-score con media/desviacion movil CENTRADA dentro de cada grupo.

    Usa `min_samples = max(3, window/2)` para no requerir la ventana
    completa al inicio/fin de la serie.
    """
    mean = (
        pl.col(metric)
        .rolling_mean(window_size=window, center=True,
                      min_samples=max(3, window // 2))
        .over(group_keys)
    )
    std = (
        pl.col(metric)
        .rolling_std(window_size=window, center=True,
                     min_samples=max(3, window // 2))
        .over(group_keys)
    )
    score = _safe_div(pl.col(metric) - mean, std)
    return lf.with_columns(
        score.alias(f"{metric}__roll_z"),
        (score.abs() > thr).alias(f"{metric}__is_roll"),
    )


def flag_seasonal(lf: pl.LazyFrame, group_keys: list[str], metric: str,
                  thr: float = SEASONAL_THRESHOLD) -> pl.LazyFrame:
    """Anomalia estacional: compara contra la mediana/MAD de la MISMA
    semana-del-anio dentro del mismo grupo (departamento)."""
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
def flag_isolation_forest(df: pl.DataFrame, group_keys: list[str],
                          metrics: list[str],
                          contamination: float = IFOREST_CONTAMINATION
                          ) -> pl.DataFrame:
    """Aplica un IsolationForest *por grupo* sobre el vector de metricas.

    - n_estimators=200, random_state=42, n_jobs=-1
    - Grupos con < 30 observaciones se omiten (no hay suficiente data).
    - Devuelve dos columnas: `is_iforest` (bool) y `iforest_score` (float,
      mayor = mas anomalo, basado en -score_samples).
    """
    flags, scores = [], []
    for _, sub in df.group_by(group_keys, maintain_order=True):
        X = sub.select(metrics).fill_null(0.0).to_numpy()
        if X.shape[0] < 30:
            flags.extend([False] * X.shape[0])
            scores.extend([0.0] * X.shape[0])
            continue
        iso = IsolationForest(n_estimators=200, contamination=contamination,
                              random_state=42, n_jobs=-1)
        iso.fit(X)
        flags.extend((iso.predict(X) == -1).tolist())
        scores.extend((-iso.score_samples(X)).tolist())
    return df.with_columns(
        pl.Series("iforest_score", scores),
        pl.Series("is_iforest", flags),
    )


# --------------------------------------------------------------------------- #
# Pipeline por tabla
# --------------------------------------------------------------------------- #
def analyze(name: str, cfg: dict) -> dict:
    """Pipeline completo para una tabla:

    1. Deduplica `(grupo, week_start)` tomando `max()` (la raw replica
       el mismo total 12+ veces por sub-ubigeo y eso satura la varianza).
    2. Aplica los 4 metodos univariados a cada metrica.
    3. Aplica Isolation Forest sobre el vector multivariado.
    4. Persiste resumen, detalle (filas con >=1 flag) y devuelve dict
       con resultados para los reportes/figuras.
    """
    log(f"--- Analizando '{name}' ({cfg['path'].name}) ---")
    lf = pl.scan_parquet(cfg["path"])

    # (1) Dedup por (grupo, week_start). max() preserva la magnitud.
    agg_cols = cfg["metrics"] + ["ano", "semana"]
    lf = (
        lf.group_by(cfg["group_keys"] + ["week_start"], maintain_order=False)
        .agg([pl.col(c).max() for c in agg_cols])
        .sort(cfg["group_keys"] + ["week_start"])
    )

    # (2) Univariados
    for m in cfg["metrics"]:
        lf = flag_iqr(lf, cfg["group_keys"], m)
        lf = flag_mad(lf, cfg["group_keys"], m)
        lf = flag_rolling_z(lf, cfg["group_keys"], m)
        lf = flag_seasonal(lf, cfg["group_keys"], m)

    df = lf.collect()
    log(f"   filas analizadas: {df.height:,}  |  columnas: {df.width}")

    # (3) Multivariado
    df = flag_isolation_forest(df, cfg["group_keys"], cfg["metrics"])

    # (4) Persistencia
    flag_cols = [c for c in df.columns
                 if c.startswith(tuple(cfg["metrics"])) and "__is_" in c]
    flag_cols.append("is_iforest")

    summary = (
        df.group_by(cfg["group_keys"], maintain_order=True)
        .agg([pl.col(c).sum().cast(pl.Int64).alias(c) for c in flag_cols]
             + [pl.len().alias("n_obs")])
    )
    summary.write_csv(REPORTS_DIR / f"outliers_summary_{name}.csv")

    any_flag = pl.fold(acc=pl.lit(False), function=lambda a, b: a | b,
                       exprs=[pl.col(c) for c in flag_cols])
    detail = df.filter(any_flag).select(
        cfg["group_keys"] + ["ano", "semana", "week_start"]
        + cfg["metrics"] + flag_cols + ["iforest_score"]
    )
    detail.write_parquet(REPORTS_DIR / f"outliers_detail_{name}.parquet")
    log(f"   detalle: {detail.height:,} filas con >=1 flag")

    return {"name": name, "df": df, "summary": summary,
            "flag_cols": flag_cols, "cfg": cfg}


# --------------------------------------------------------------------------- #
# Visualizacion
# --------------------------------------------------------------------------- #
def plot_top(result: dict, metric: str, top_n: int = 4,
             label_top_k: int = 8) -> None:
    """Genera un PNG por cada uno de los `top_n` departamentos con mas
    flags. Etiqueta numericamente los `label_top_k` picos mas altos de
    cada metodo (no satura el grafico)."""
    name, df, cfg = result["name"], result["df"], result["cfg"]
    group_keys = cfg["group_keys"]
    flag_col = f"{metric}__is_mad"
    if flag_col not in df.columns:
        return

    top = (
        df.group_by(group_keys).agg(pl.col(flag_col).sum().alias("n"))
        .sort("n", descending=True).head(top_n)
        .get_column(group_keys[0]).to_list()
    )
    colors = {"IQR": "#ff7f0e", "MAD": "#2ca02c", "Rolling": "#d62728",
              "Seasonal": "#9467bd", "IForest": "#8c564b"}

    for dept in top:
        sub = df.filter(pl.col(group_keys[0]) == dept).sort("week_start")
        x = sub.get_column("week_start").to_list()
        y = sub.get_column(metric).to_list()

        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(x, y, color="#1f77b4", lw=0.9, alpha=0.75, label=metric)

        masks = {label: sub.get_column(col).to_list() for label, col in [
            ("IQR", f"{metric}__is_iqr"), ("MAD", f"{metric}__is_mad"),
            ("Rolling", f"{metric}__is_roll"),
            ("Seasonal", f"{metric}__is_seas"),
            ("IForest", "is_iforest"),
        ]}

        labels_to_show: list[tuple] = []
        for label, mask in masks.items():
            xs = [xi for xi, m in zip(x, mask) if m]
            ys = [yi for yi, m in zip(y, mask) if m]
            if not xs:
                continue
            ax.scatter(xs, ys, s=28, color=colors[label], alpha=0.85,
                       edgecolor="black", linewidth=0.4, zorder=3,
                       label=f"{label} ({len(xs)})")
            order = sorted(range(len(ys)), key=lambda i: ys[i],
                           reverse=True)[:label_top_k]
            for i in order:
                labels_to_show.append((xs[i], ys[i], colors[label]))

        for xi, yi, color in labels_to_show:
            ax.annotate(f"{yi:,.0f}", xy=(xi, yi), xytext=(0, 8),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=7, color=color, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec=color, lw=0.5, alpha=0.85))

        ax.set_title(f"Outliers - {dept} - {metric}", fontsize=12,
                     fontweight="bold")
        ax.set_xlabel("Semana"); ax.set_ylabel(metric)
        ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
        ax.grid(alpha=0.3); fig.tight_layout()

        safe = "".join(c if c.isalnum() else "_" for c in dept)[:40]
        out = FIG_DIR / f"outliers_{name}_{safe}_{metric}.png"
        fig.savefig(out, dpi=130); plt.close(fig)
        log(f"   figura -> {out.relative_to(BASE_DIR)}")


# --------------------------------------------------------------------------- #
# Reporte ejecutivo
# --------------------------------------------------------------------------- #
def executive_report(results: list[dict]) -> None:
    """Texto plano con: % de flags por metodo y top-5 grupos por IForest."""
    lines = ["=" * 78,
             "REPORTE EJECUTIVO - DETECCION DE OUTLIERS Y ANOMALIAS",
             "=" * 78, ""]
    for r in results:
        name, df, summary, flag_cols = r["name"], r["df"], r["summary"], r["flag_cols"]
        lines.append(f"## Tabla: {name}    filas={df.height:,}")
        totals = {c: int(df.get_column(c).sum()) for c in flag_cols}
        for c, v in sorted(totals.items(), key=lambda kv: -kv[1]):
            pct = 100 * v / df.height
            lines.append(f"   {c:<55s} {v:>8,}  ({pct:5.2f}%)")
        lines.append("   Top-5 grupos por IForest:")
        for row in (summary.sort("is_iforest", descending=True).head(5)
                    .select(r["cfg"]["group_keys"] + ["is_iforest", "n_obs"])
                    .iter_rows(named=True)):
            grp = " / ".join(str(row[k]) for k in r["cfg"]["group_keys"])
            lines.append(f"      - {grp:<35s} {row['is_iforest']:>5} / {row['n_obs']:>5}")
        lines.append("")
    txt = "\n".join(lines)
    out = REPORTS_DIR / "outliers_report.txt"
    out.write_text(txt, encoding="utf-8")
    print("\n" + txt)
    log(f"reporte ejecutivo -> {out.relative_to(BASE_DIR)}")


def main() -> None:
    log(f"Polars {pl.__version__}  |  threads={pl.thread_pool_size()}")
    t0 = time.time()
    results = []
    for name, cfg in TARGETS.items():
        if not cfg["path"].exists():
            log(f"!! No existe {cfg['path']}, se omite.")
            continue
        results.append(analyze(name, cfg))
    for r in results:
        primary = ("cases_total" if "cases_total" in r["cfg"]["metrics"]
                   else r["cfg"]["metrics"][0])
        plot_top(r, primary, top_n=4)
    executive_report(results)
    log(f"Listo en {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
