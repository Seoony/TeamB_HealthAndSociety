# `src/outliers/` — Detección de Outliers y Anomalías

Módulo responsable de identificar valores atípicos en las series semanales
de IRAs y Neumonía por departamento. Estos picos pueden ser brotes reales
o errores de captura: separarlos es necesario antes del modelado predictivo.

## Archivos

| Archivo | Función |
|---|---|
| `__init__.py` | Marca el paquete. |
| `detection.py` | Pipeline principal con 5 métodos de detección + visualización. |

## Métodos implementados

| Método | Tipo | Umbral |
|---|---|---|
| **IQR** (Tukey) | Univariado | `k = 1.5` |
| **Modified Z-score (MAD)** | Univariado, robusto | `|z| > 3.5` |
| **Rolling Z-score** | Temporal local (ventana 13 semanas) | `|z| > 3.0` |
| **Seasonal** | Compara con misma semana-del-año | `|z| > 3.5` |
| **Isolation Forest** | Multivariado | `contamination = 1%` |

## Cómo ejecutar

Desde la raíz del proyecto, con el venv activado:

```powershell
.\Scripts\python.exe -X utf8 -m src.outliers.detection
```

Tiempo aproximado: **~30 s** sobre las 2 tablas semanales (~120k filas
antes del dedup, ~60k después). Usa todos los hilos disponibles
(`pl.thread_pool_size()`).

## Outputs

Todo se escribe en `reports/outliers/`:

- `outliers_report.txt` — resumen ejecutivo.
- `outliers_summary_<tabla>.csv` — conteo de flags por departamento × método.
- `outliers_detail_<tabla>.parquet` — todas las filas con ≥1 flag.
- `figures/outliers_<tabla>_<dept>_<metrica>.png` — series con outliers
  marcados por método y el valor numérico en los 8 picos más altos de
  cada método.

## Tablas analizadas

- `data/raw/pneumonia_weekly_incidence_dept.parquet` (incidencia normalizada
  → comparable entre departamentos).
- `data/raw/iras_weekly_dept.parquet` (conteos absolutos).

> Nota: el pipeline deduplica internamente por `(department, week_start)`
> con `max()`. La tabla raw repite cada total 12+ veces por sub-ubigeo y
> eso satura la varianza local del rolling Z-score.
