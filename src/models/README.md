# `src/models/` — Predicción (Fases 2–4)

## Fase 2 — tabla de modelado

```bash
python -m src.models.dataset
```

Usa: `data/processed/modeling_iras_dept.parquet`

## Fase 3 — evaluación

```bash
python -m src.models.evaluate
```

| Archivo | Rol |
|---|---|
| `protocol.py` | Target, horizonte, splits, features |
| `dataset.py` | Lags + `y` + split |
| `forecasts.py` | Naive, estacional, SARIMA, XGBoost |
| `evaluate.py` | Walk-forward + MAE/RMSE |

**Usar:**

- `data/processed/phase3_predictions.parquet`
- `data/processed/phase3_metrics.parquet`

Modelos: `yhat_naive`, `yhat_seasonal`, `yhat_sarima`, `yhat_xgb`.  
`mae_skill_vs_naive` > 0 significa que gana al naive.

COVID (`holdout_covid`) no se usa para entrenar XGBoost ni para puntuar.

## Fase 4 — robustez

```bash
python -m src.models.robustness
```

Requiere Fase 3. Salidas:

- `phase4_intervals.parquet` — intervalo 90% (conformal, calibrado en val)
- `phase4_coverage.parquet` — cobertura y ancho
- `phase4_sensitivity.parquet` — COVID, silencios, sin `flag_silence`
- `phase4_transfer.parquet` — modelo Arequipa aplicado a otros pilotos
