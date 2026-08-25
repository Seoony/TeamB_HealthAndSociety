"""Contrato de las tablas limpias (`data/processed/*_cleaned.parquet`).

Una fila = un lugar × una semana epidemiológica.
Los modelos y una futura API deben leer estas columnas, no `data/raw/`.
"""

from __future__ import annotations

from typing import Literal, TypedDict

GeoLevel = Literal["department", "province"]

FLAG_COLS: tuple[str, ...] = (
    "flag_death_no_hosp",
    "flag_hosp_corrected",
    "flag_silence",
)

KEY_DEPT: tuple[str, ...] = (
    "geo_level",
    "department",
    "ubigeo",
    "ano",
    "semana",
    "week_start",
)

KEY_PROV: tuple[str, ...] = (
    "geo_level",
    "department",
    "province",
    "ubigeo",
    "ano",
    "semana",
    "week_start",
)


class CleaningSpec(TypedDict):
    raw: str
    geo_level: GeoLevel
    key: list[str]
    metrics: list[str]
    passthrough: list[str]


CLEANING_PLAN: dict[str, CleaningSpec] = {
    "iras_weekly_dept": {
        "raw": "iras_weekly_dept.parquet",
        "geo_level": "department",
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
        "geo_level": "province",
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
        "geo_level": "department",
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
        "geo_level": "province",
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


def analysis_columns(spec: CleaningSpec) -> list[str]:
    """Orden estable para análisis: claves → métricas → flags."""
    keys = KEY_PROV if spec["geo_level"] == "province" else KEY_DEPT
    return [*keys, *spec["metrics"], *FLAG_COLS]


def required_raw_columns(spec: CleaningSpec) -> list[str]:
    """Columnas que se leen del parquet raw (sin duplicar)."""
    seen: set[str] = set()
    ordered: list[str] = []
    for col in [*spec["key"], *spec["metrics"], *spec["passthrough"]]:
        if col not in seen:
            seen.add(col)
            ordered.append(col)
    return ordered
