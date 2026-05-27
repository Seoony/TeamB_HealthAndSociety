# `src/integrity/` — Validación y Limpieza de Integridad

Módulo responsable de verificar la calidad de los datos extraídos
(`data/raw/`) y de generar versiones limpias (`data/processed/`) listas
para el modelado.

## Archivos

| Archivo | Función |
|---|---|
| `__init__.py` | Marca el paquete. |
| `validation.py` | Ejecuta los **checks** y produce reportes diagnósticos. |
| `cleaning.py` | Aplica las **correcciones** derivadas del diagnóstico. |

## Pipeline recomendado

```
data/raw/*.parquet
       │
       ▼
src.integrity.validation   ──►  reports/integrity/*.{txt,csv,parquet,png}
       │
       ▼
src.integrity.cleaning     ──►  data/processed/*_cleaned.parquet
                                reports/integrity/cleaning_report.txt
                                reports/integrity/silence_by_province.csv
```

## 1) `validation.py` — Checks de calidad

Detecta sobre `hosp_*` y `deaths_*`:

1. **Nulos** por columna y por departamento.
2. **Valores negativos** (imposibles).
3. **Inconsistencia de estratos**: `under5 + 60plus > total`.
4. **Coherencia lógica**: `hosp > cases`, `deaths > cases`, `deaths > hosp`.
5. **Duplicados** por clave natural (`ubigeo + week_start`).
6. **Rangos**: año ∈ [1999, 2026], semana ∈ [1, 53].
7. **Cobertura temporal**: grupos con < 95 % de semanas.
8. **Silencios**: `cases > 0` pero `hosp = 0` y `deaths = 0`.

### Cómo ejecutar

```powershell
.\Scripts\python.exe -X utf8 -m src.integrity.validation
```

Tiempo aproximado: **~25 s** sobre las 5 tablas (2 M + 700 k + 730 k + 173 k + 94 k filas).

### Outputs (en `reports/integrity/`)

- `integrity_report.txt` — resumen ejecutivo.
- `integrity_overview.csv` — matriz tabla × check.
- `integrity_summary_<tabla>.csv` — nulos por columna.
- `integrity_gaps_<tabla>.csv` — cobertura temporal por departamento.
- `integrity_issues_<tabla>.parquet` — filas con ≥1 issue.
- `figures/integrity_issues_<tabla>.png` — barras (log) con cada hallazgo.
- `figures/integrity_nulls_<tabla>.png` — heatmap dept × columna.
- `figures/integrity_timeline_<tabla>.png` — evolución del % de nulos por año.

## 2) `cleaning.py` — Correcciones automáticas

Implementa las 4 recomendaciones del diagnóstico:

| ID | Acción | Detalle |
|---|---|---|
| **R1** | Deduplicar weekly | `group_by(key).agg(max())` colapsa réplicas por sub-ubigeo. |
| **R2** | Reportar silencios | `silence_by_province.csv` con % global y pendiente OLS por año. |
| **R3** | Marcar `deaths > hosp` | Agrega `flag_death_no_hosp` (no elimina filas). |
| **R4** | Corregir `hosp > cases` | Clamp `cases = max(cases, hosp)` + `flag_hosp_corrected`. |

### Cómo ejecutar

```powershell
.\Scripts\python.exe -X utf8 -m src.integrity.cleaning
```

Tiempo aproximado: **~3 s**.

### Outputs

- `data/processed/<tabla>_cleaned.parquet` — tablas listas para modelado.
- `reports/integrity/cleaning_report.txt` — resumen del proceso.
- `reports/integrity/silence_by_province.csv` — diagnóstico de sub-registro.

### Reducción de duplicados aplicada

| Tabla | Filas raw | Filas limpias | Removidas |
|---|---:|---:|---:|
| `iras_weekly_dept` | 93 924 | 31 225 | 62 699 |
| `iras_weekly_prov` | 731 346 | 243 193 | 488 153 |
| `pneumonia_dept` | 172 824 | 28 726 | 144 098 |
| `pneumonia_prov` | 701 862 | 233 389 | 468 473 |

## Decisiones de diseño

- **No tocamos `data/raw/`**. La limpieza siempre va a `data/processed/`.
- **Marcamos en vez de eliminar.** Las filas con `deaths > hosp` pueden
  ser muertes domiciliarias legítimas: descartarlas sesgaría el modelo
  de mortalidad. Quedan accesibles vía `flag_death_no_hosp`.
- **Clamp suave para `hosp > cases`.** Lo más conservador: forzar
  `cases = hosp` (no se puede hospitalizar sin caso). Auditable por
  `flag_hosp_corrected`.
