"""Predictores de Fase 3: naive, naive estacional, SARIMA, XGBoost.

Ninguno usa información posterior al origen t.
"""

from __future__ import annotations

from collections.abc import Iterable
from warnings import catch_warnings, simplefilter

import numpy as np
import pandas as pd
import polars as pl
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from src.models import protocol as P


def feature_frame(
    df: pl.DataFrame,
    columns: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    cols = list(columns or P.XGB_FEATURES)
    pdf = df.select(cols).to_pandas()
    if "department" in pdf.columns:
        pdf["department"] = pdf["department"].astype("category")
    for col in P.FEATURE_FLAGS:
        if col in pdf.columns:
            pdf[col] = pdf[col].fillna(False).astype("int8")
    return pdf


def clip_count(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=float), 0.0)


def add_naive_forecasts(frame: pl.DataFrame) -> pl.DataFrame:
    """Persistencia (nivel en t) y estacional (misma semana del año del target)."""
    return (
        frame.sort(["department", "week_start"])
        .with_columns(
            yhat_naive=pl.col(P.TARGET_COL).fill_null(0.0),
            yhat_seasonal=pl.coalesce(
                pl.col(P.TARGET_COL).shift(52 - P.HORIZON_WEEKS).over("department"),
                pl.col(P.TARGET_COL),
            ).fill_null(0.0),
        )
    )


def fit_xgb(
    train: pl.DataFrame,
    eval_set: pl.DataFrame | None = None,
    feature_cols: tuple[str, ...] | None = None,
) -> XGBRegressor:
    cols = feature_cols or P.XGB_FEATURES
    x_train = feature_frame(train, cols)
    y_train = train["y"].to_numpy()
    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=4,
        objective="reg:squarederror",
        tree_method="hist",
        enable_categorical=True,
        random_state=P.RANDOM_SEED,
        n_jobs=-1,
    )
    if eval_set is not None and eval_set.height >= 50:
        model.set_params(early_stopping_rounds=30)
        model.fit(
            x_train,
            y_train,
            eval_set=[(feature_frame(eval_set, cols), eval_set["y"].to_numpy())],
            verbose=False,
        )
    else:
        model.fit(x_train, y_train, verbose=False)
    return model


def predict_xgb(
    model: XGBRegressor,
    frame: pl.DataFrame,
    feature_cols: tuple[str, ...] | None = None,
) -> np.ndarray:
    return clip_count(model.predict(feature_frame(frame, feature_cols)))


def walk_forward_xgb(
    frame: pl.DataFrame,
    pred_split: str,
    history_splits: Iterable[str],
    refit_every: int = P.REFIT_EVERY_WEEKS,
    feature_cols: tuple[str, ...] | None = None,
    train_departments: Iterable[str] | None = None,
    pred_departments: Iterable[str] | None = None,
    yhat_name: str = "yhat_xgb",
) -> pl.DataFrame:
    """Reentrena cada `refit_every` orígenes con outcomes ya observados."""
    pred = frame.filter(pl.col("split") == pred_split)
    if pred_departments is not None:
        pred = pred.filter(pl.col("department").is_in(list(pred_departments)))
    pred = pred.sort(["week_start", "department"])
    if pred.is_empty():
        return pred.with_columns(pl.lit(None, dtype=pl.Float64).alias(yhat_name))

    pool = frame
    if train_departments is not None:
        pool = frame.filter(pl.col("department").is_in(list(train_departments)))

    origins = pred.get_column("week_start").unique().sort().to_list()
    history = set(history_splits)
    chunks: list[pl.DataFrame] = []
    model: XGBRegressor | None = None
    last_fit = -refit_every
    cols = feature_cols or P.XGB_FEATURES

    for i, origin in enumerate(origins):
        if model is None or (i - last_fit) >= refit_every:
            known = pool.filter(
                pl.col("split").is_in(list(history))
                | (
                    (pl.col("split") == pred_split)
                    & (pl.col("target_week_start") <= origin)
                )
            )
            if known.height < 80:
                known = pool.filter(pl.col("split").is_in(list(history)))
            model = fit_xgb(known, feature_cols=cols)
            last_fit = i
        batch = pred.filter(pl.col("week_start") == origin)
        chunks.append(
            batch.with_columns(
                pl.Series(predict_xgb(model, batch, cols)).alias(yhat_name)
            )
        )

    return pl.concat(chunks, how="vertical")


def _prepare_series(dept_frame: pl.DataFrame) -> pd.Series:
    series = (
        dept_frame.sort("week_start")
        .select("week_start", P.TARGET_COL)
        .to_pandas()
        .drop_duplicates("week_start")
        .set_index("week_start")[P.TARGET_COL]
        .astype(float)
        .sort_index()
        .asfreq("W-MON")
        .interpolate(limit=2)
    )
    series = series.dropna()
    if series.size:
        series.index = pd.DatetimeIndex(series.index, freq="W-MON")
    return series


def _sarima_forecast(history: pd.Series, steps: int) -> float:
    if history.size < 24:
        return float(history.iloc[-1]) if history.size else 0.0
    try:
        with catch_warnings():
            simplefilter("ignore")
            result = SARIMAX(
                history,
                order=(1, 1, 1),
                seasonal_order=(0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=40, method="lbfgs")
        forecast = result.forecast(steps=steps)
        return float(clip_count(np.asarray([forecast.iloc[-1]]))[0])
    except Exception:
        return float(history.iloc[-1])


def walk_forward_sarima(
    frame: pl.DataFrame,
    pred_split: str,
    refit_every: int = P.REFIT_EVERY_WEEKS,
) -> pl.DataFrame:
    """Univariado por departamento: pronóstico a HORIZON semanas desde t."""
    pred = frame.filter(pl.col("split") == pred_split)
    if pred.is_empty():
        return pred.with_columns(yhat_sarima=pl.lit(None, dtype=pl.Float64))

    pieces: list[pl.DataFrame] = []
    depts = pred.get_column("department").unique().sort().to_list()
    for i, dept in enumerate(depts, start=1):
        print(f"    SARIMA {i}/{len(depts)} {dept}", flush=True)
        dept_all = frame.filter(pl.col("department") == dept)
        series = _prepare_series(dept_all)
        batch = pred.filter(pl.col("department") == dept).sort("week_start")
        yhats = [
            _sarima_forecast(
                series.loc[series.index <= pd.Timestamp(origin)],
                P.HORIZON_WEEKS,
            )
            for origin in batch.get_column("week_start").to_list()
        ]
        pieces.append(batch.with_columns(yhat_sarima=pl.Series(yhats, dtype=pl.Float64)))
    return pl.concat(pieces, how="vertical")
