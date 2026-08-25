# `src/integrity/` — Integridad de datos

El **corazón** es la limpieza: `data/raw/` → `data/processed/*_cleaned.parquet`.
Validación y reportes son jobs aparte (no se mezclan con el núcleo).

## Núcleo

```text
data/raw/*.parquet
        │
        ▼
src.integrity.cleaning   ──►  data/processed/*_cleaned.parquet
```

```bash
python -m src.integrity.cleaning
python -m src.integrity.cleaning --table iras_weekly_dept
python -m src.pipeline
```

Reglas en la tabla (no en un PDF):

| ID | Columna / acción | Uso en análisis |
|---|---|---|
| R1 | Dedup por lugar × semana (`max` en métricas) | Serie semanal sin réplicas |
| R3 | `flag_death_no_hosp` | No filtrar a ciegas (muerte domiciliaria) |
| R4 | `flag_hosp_corrected` + clamp de `cases` | Auditar corrección |
| R2 | `flag_silence` | Subregistro / casos sin hosp ni muertes |

Contrato: `schema.py`. Cargar para modelar: `load_cleaned("iras_weekly_dept")`.

## Aparte (no el corazón)

```bash
python -m src.integrity.validation   # diagnóstico raw → reports/integrity/
python -m src.integrity.reports      # silence_by_province.csv
python -m src.outliers.detection
python -m src.analysis.hosp_deaths_analysis
```
