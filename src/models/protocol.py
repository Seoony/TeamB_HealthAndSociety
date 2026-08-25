"""Protocolo de predicción (Fase 2). No cambiar después de ver métricas.

Pregunta
--------
¿Cuántas hospitalizaciones en 60+ por IRA habrá en un departamento
en las próximas HORIZON_WEEKS semanas?

Split por fecha del *target* (semana t+h), no del origen t:
así ningún outcome de 2019 entra en train.
"""

from __future__ import annotations

from typing import Final

SOURCE_TABLE: Final = "iras_weekly_dept"
TARGET_COL: Final = "hosp_60plus"
HORIZON_WEEKS: Final = 4

LAGS: Final[tuple[int, ...]] = (1, 2, 4, 8)
ROLLING_WINDOWS: Final[tuple[int, ...]] = (4, 13)
LAG_SOURCES: Final[tuple[str, ...]] = (
    "hosp_60plus",
    "deaths_60plus",
    "cases_total",
)

YEAR_STUDY_START: Final = 2006
YEAR_TRAIN_END: Final = 2018
YEAR_VAL: Final = 2019
YEAR_COVID: Final[tuple[int, ...]] = (2020, 2021)
YEAR_TEST: Final[tuple[int, ...]] = (2022, 2023)

PILOT_DEPARTMENTS: Final[tuple[str, ...]] = (
    "AREQUIPA",
    "TACNA",
    "MOQUEGUA",
    "TUMBES",
    "LIMA",
)

# Semanas epidemiológicas de mayor carga invernal (Perú). Uso en Fase 3.
WINTER_WEEKS: Final[tuple[int, int]] = (20, 35)

SPLIT_TRAIN: Final = "train"
SPLIT_VAL: Final = "val"
SPLIT_TEST: Final = "test"
SPLIT_COVID: Final = "holdout_covid"
SPLIT_DROP: Final = "drop"

OUTPUT_NAME: Final = "modeling_iras_dept.parquet"

FEATURE_LAGS: Final[tuple[str, ...]] = tuple(
    f"{col}_lag{lag}" for col in LAG_SOURCES for lag in LAGS
)
FEATURE_ROLLING: Final[tuple[str, ...]] = tuple(
    f"{TARGET_COL}_ma{w}" for w in ROLLING_WINDOWS
)
FEATURE_FLAGS: Final[tuple[str, ...]] = (
    "flag_silence",
    "flag_death_no_hosp",
    "flag_hosp_corrected",
    "flag_calendar_gap",
    "flag_pop_carried_forward",
)
FEATURE_LEVEL: Final[tuple[str, ...]] = (
    "ano",
    "semana",
    TARGET_COL,
    "deaths_60plus",
    "cases_total",
    "population",
    "hosp_60plus_per_100k",
)

REQUIRED_FOR_MODEL: Final[tuple[str, ...]] = (
    "y",
    TARGET_COL,
    *FEATURE_LAGS,
    *FEATURE_ROLLING,
)

# Fase 3 — evaluación
XGB_FEATURES: Final[tuple[str, ...]] = (
    "department",
    *FEATURE_LEVEL,
    *FEATURE_LAGS,
    *FEATURE_ROLLING,
    *FEATURE_FLAGS,
)
REFIT_EVERY_WEEKS: Final = 4
PHASE3_PRED_NAME: Final = "phase3_predictions.parquet"
PHASE3_METRICS_NAME: Final = "phase3_metrics.parquet"
RANDOM_SEED: Final = 42

# Fase 4 — robustez
INTERVAL_ALPHA: Final = 0.10  # intervalo 90% (colas 5% / 5%)
TRANSFER_SOURCE: Final = "AREQUIPA"
TRANSFER_FEATURES: Final[tuple[str, ...]] = tuple(
    c for c in XGB_FEATURES if c != "department"
)
PHASE4_INTERVALS_NAME: Final = "phase4_intervals.parquet"
PHASE4_COVERAGE_NAME: Final = "phase4_coverage.parquet"
PHASE4_SENSITIVITY_NAME: Final = "phase4_sensitivity.parquet"
PHASE4_TRANSFER_NAME: Final = "phase4_transfer.parquet"
