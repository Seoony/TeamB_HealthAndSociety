"""Reportes de integridad (capa de presentación, no del corazón).

No se ejecuta junto a la limpieza. Queda para una UI o un job aparte:

    python -m src.integrity.reports
"""

from __future__ import annotations

import polars as pl

from src.config import config
from src.integrity.cleaning import load_cleaned, silence_mask

REPORTS_DIR = config.BASE_DIR / "reports" / "integrity"


def silence_by_province(
    frame: pl.DataFrame | pl.LazyFrame,
    year_col: str = "ano",
) -> pl.DataFrame:
    """% de semanas con casos>0 y sin hosp ni muertes, por provincia.

    `slope_pct_per_year`: >0 el subregistro empeora; <0 mejora.
    """
    lf = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
    marked = lf.with_columns(silence_mask().alias("_silence"))

    global_pct = marked.group_by(["department", "province"]).agg(
        pct_silence=(pl.col("_silence").sum() / pl.len() * 100).round(3),
        n_obs=pl.len(),
    )

    yearly = (
        marked.group_by(["department", "province", year_col])
        .agg(pct=(pl.col("_silence").sum() / pl.len() * 100))
        .collect()
    )

    def _slope(sub: pl.DataFrame) -> float:
        if sub.height < 2:
            return 0.0
        x = sub.get_column(year_col).to_numpy().astype(float)
        y = sub.get_column("pct").to_numpy().astype(float)
        x_mean = x.mean()
        y_mean = y.mean()
        denom = ((x - x_mean) ** 2).sum()
        if denom == 0:
            return 0.0
        return float(((x - x_mean) * (y - y_mean)).sum() / denom)

    slopes = (
        yearly.group_by(["department", "province"], maintain_order=True)
        .agg(pl.struct([year_col, "pct"]).alias("_records"))
        .with_columns(
            slope_pct_per_year=pl.col("_records")
            .map_elements(
                lambda recs: _slope(pl.DataFrame(recs.to_list())),
                return_dtype=pl.Float64,
            )
            .round(4)
        )
        .drop("_records")
    )

    return (
        global_pct.collect()
        .join(slopes, on=["department", "province"], how="left")
        .sort("pct_silence", descending=True)
    )


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cleaned = load_cleaned("iras_weekly_prov")
    silence = silence_by_province(cleaned)
    out = REPORTS_DIR / "silence_by_province.csv"
    silence.write_csv(out)
    print(f"escrito {out.relative_to(config.BASE_DIR)} ({silence.height} provincias)")


if __name__ == "__main__":
    main()
