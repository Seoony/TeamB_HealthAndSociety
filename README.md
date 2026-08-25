# Proyecto UNSA & Oklahoma: Salud y Sociedad (Equipo B)

Análisis predictivo de **hospitalizaciones por IRAs en adultos de 60+** (Perú), colaboración **UNSA** / **University of Oklahoma**.

Este repositorio es el **núcleo Python**: datos, limpieza, modelado y evaluación.

## Integrantes

* **Calienes Rodríguez, Ricardo Fabrizio** (Docente encargado / Líder Team B)
* **Rivas Chire, Anthony Juancarlo** (Líder de Equipo)
* **Haytara Tonconi, Alex Antonio**
* **Quispe Ttito, Juan Carlos**
* **Quispe Huacho, Rodolfo Robert** (El dev mas gozu)
* **Zapana Romero, Pedro Luis Christian**

## Qué hace este repo

1. Descarga tablas de OSCER → `data/raw/`
2. Limpia → `data/processed/*_cleaned.parquet`
3. Arma el dataset de modelado (`y` = hosp. 60+ a **4 semanas**)
4. Evalúa naive / ARIMA / XGBoost (walk-forward)
5. Robustez (intervalos 90%, COVID, transferencia Arequipa → otros)

No hay `main.py`. No hay API. Los reportes PNG/TXT son jobs opcionales.

## Requisitos

- Python **3.12+** (Ubuntu: `sudo apt install python3.12-venv` si `python3 -m venv` falla)
- Credenciales OSCER **o** parquet ya copiados en `data/raw/`

## Arranque (Linux)

```bash
cd TeamB_HealthAndSociety
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mkdir -p data/raw data/processed data/external notebooks models
cp .env.example .env
```

Edita `.env` con usuario, contraseña, host, puerto y `DB_NAME`. El `.env` **no se sube a Git**.

Activar el venv en cada terminal: `source venv/bin/activate`.

## Datos

```bash
python -m src.data_loader     # OSCER → data/raw/*.parquet
python -m src.pipeline        # limpieza → data/processed/*_cleaned.parquet
```

Si un compañero ya te pasó los parquet, colócalos en `data/raw/` y omite `data_loader`.

Limpieza de una sola tabla:

```bash
python -m src.integrity.cleaning --table iras_weekly_dept
```

## Modelado y evaluación

```bash
python -m src.models.dataset       # modeling_iras_dept.parquet
python -m src.models.evaluate      # ~9 min → phase3_*.parquet
python -m src.models.robustness    # ~1 min → phase4_*.parquet
```

**Usar en análisis / paper:**

- `data/processed/modeling_iras_dept.parquet`
- `data/processed/phase3_predictions.parquet`
- `data/processed/phase3_metrics.parquet`
- `data/processed/phase4_*.parquet`

Protocolo (no cambiar después de ver métricas): `src/models/protocol.py`.

Cargar:

```python
from src.models.dataset import load_modeling
df = load_modeling()
train = df.filter(pl.col("split") == "train")
```

Jobs opcionales (no forman el núcleo):

```bash
python -m src.integrity.validation
python -m src.outliers.detection
python -m src.analysis.hosp_deaths_analysis
```

## Hallazgo (resumen)

Pronóstico a 4 semanas de hospitalizaciones 60+ por IRA, **por departamento**.

- **ARIMA(1,1,1)** walk-forward gana al naive en test 2022–23 (~+15% skill MAE).
- **XGBoost** ayuda en Lima y empeora en Tumbes (conteos bajos).
- El naive estacional se rompe post-COVID.
- Un modelo entrenado solo en **Arequipa no transfiere** a Tacna, Moquegua, Tumbes ni Lima.
- Los intervalos 90% (conformal en 2019) **subcubren** en 2022–23.

## Estructura

```text
src/config.py
src/data_loader.py
src/pipeline.py                 # solo limpieza
src/integrity/cleaning.py       # corazón: raw → processed
src/integrity/validation.py     # reportes (aparte)
src/models/protocol.py
src/models/dataset.py
src/models/evaluate.py
src/models/robustness.py
data/raw/                       # gitignored
data/processed/                 # gitignored
.env                            # gitignored
```

## Investigación

Colaboración UNSA–OU para análisis de salud pública (IRAs / neumonía, Perú, foco Arequipa).
Los resultados están pensados para integrarse en PanViz y en manuscritos.
El modelado implementado hoy es **ARIMA + XGBoost** sobre series departamentales semanales de IRA 60+; LSTM/provincia/menores de 5 no están en este pipeline.
