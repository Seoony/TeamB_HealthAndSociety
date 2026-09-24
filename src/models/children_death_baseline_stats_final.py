from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import time

warnings.filterwarnings("ignore")

inicio = time.perf_counter()

# SETTINGS

ROOT_DIR = Path(__file__).resolve().parents[2]

IRA_PATH = ROOT_DIR / "data" / "raw" / "iras_data_raw_temp.parquet"
POP_PATH = ROOT_DIR / "data" / "raw" / "population_dept_long.parquet"

OUTPUT_DIR = ROOT_DIR / "outputs" / "children_death_baseline_stat_optimized"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REGIONS = [
    "AMAZONAS",
    "ANCASH",
    "APURIMAC",
    "AREQUIPA",
    "AYACUCHO",
    "CAJAMARCA",
    "CALLAO",
    "CUSCO",
    "HUANCAVELICA",
    "HUANUCO",
    "ICA",
    "JUNIN",
    "LA LIBERTAD",
    "LAMBAYEQUE",
    "LIMA",
    "LORETO",
    "MADRE DE DIOS",
    "MOQUEGUA",
    "PASCO",
    "PIURA",
    "PUNO",
    "SAN MARTIN",
    "TACNA",
    "TUMBES",
    "UCAYALI",
]

TARGET_COL = "defunciones_men5"

START_YEAR = 2006
END_YEAR = 2023

SMOOTH_WINDOW = 3

TRAIN_WINDOW_WEEKS = 5 * 52
FORECAST_HORIZON = 4
STEP_WEEKS = 4
SEASONAL_PERIOD = 52

FUTURE_WEEKS = 52
FUTURE_BLOCKS = FUTURE_WEEKS // FORECAST_HORIZON

USE_LOG_TARGET = True
CAP_OUTLIERS = True
CAP_QUANTILE = 0.99

ARIMA_ORDERS = [
    (1, 1, 1),
    (2, 1, 1),
    (1, 1, 2),
    (2, 1, 2),
    (3, 1, 1)
]

MODEL_COLORS = {
    "actual": "black",
    "naive": "tab:blue",
    "seasonal_naive": "tab:orange",
    "holt_winters": "tab:green",
    "arima": "tab:red",
}

BEST_MODEL_FOR_PLOTS = "holt_winters"


def display_name_for(model_key):
    names = {
        "naive": "Naive",
        "seasonal_naive": "Seasonal Naive",
        "holt_winters": "Holt-Winters",
        "arima": "ARIMA",
    }
    return names.get(model_key, model_key)


# HELPER FUNCTIONS

def transform_y(y):
    y = np.asarray(y, dtype=float)
    y = np.clip(y, 0, None)

    if USE_LOG_TARGET:
        return np.log1p(y)

    return y


def inverse_transform_y(y_transformed):
    y_transformed = np.asarray(y_transformed, dtype=float)

    if USE_LOG_TARGET:
        y = np.expm1(y_transformed)
    else:
        y = y_transformed

    return np.clip(y, 0, None)


def safe_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# LOAD DATA

def load_data():
    ira = pd.read_parquet(IRA_PATH)
    print(f"Número total de filas: {ira.shape[0]}")
    pop = pd.read_parquet(POP_PATH)

    ira.columns = ira.columns.str.lower().str.strip()
    pop.columns = pop.columns.str.lower().str.strip()

    if TARGET_COL not in ira.columns:
        raise KeyError(
            f"{TARGET_COL} not found. Available IRA columns:\n{ira.columns.tolist()}"
        )

    ira["iddpto"] = ira["iddpto"].astype(str).str.zfill(2)
    ira["departamento"] = ira["departamento"].astype(str).str.strip().str.upper()
    pop["department"] = pop["department"].astype(str).str.strip().str.upper()

    ira["ano"] = pd.to_numeric(ira["ano"], errors="coerce").astype(int)
    ira["semana"] = pd.to_numeric(ira["semana"], errors="coerce").astype(int)

    ira["date"] = pd.to_datetime(
        ira["ano"].astype(str) + "-" + ira["semana"].astype(str).str.zfill(2) + "-1",
        format="%G-%V-%u",
        errors="coerce"
    )

    ira = ira.dropna(subset=["date"]).copy()
    ira["year"] = ira["date"].dt.year.astype(int)
    ira = ira[(ira["year"] >= START_YEAR) & (ira["year"] <= END_YEAR)].copy()

    ira[TARGET_COL] = pd.to_numeric(ira[TARGET_COL], errors="coerce").fillna(0)

    pop["year"] = pd.to_numeric(pop["year"], errors="coerce").astype(int)
    pop["population"] = pd.to_numeric(pop["population"], errors="coerce")
    pop = pop.drop_duplicates(subset=["department", "year"]).copy()

    return ira, pop


def prepare_region_series(ira, pop, region_name):
    region = region_name.upper()

    ira_region = ira[ira["departamento"] == region].copy()

    weekly = (
        ira_region
        .groupby(["date", "year", "departamento"], as_index=False)[TARGET_COL]
        .sum()
    )

    weekly = weekly.merge(
        pop[["department", "year", "population"]],
        left_on=["departamento", "year"],
        right_on=["department", "year"],
        how="left"
    )
    weekly = weekly.drop(columns=["department"])

    weekly = weekly.dropna(subset=["population"]).copy()

    weekly["incidence_raw"] = (weekly[TARGET_COL] / weekly["population"]) * 100000

    if CAP_OUTLIERS:
        cap_value = weekly["incidence_raw"].quantile(CAP_QUANTILE)
        weekly["incidence_raw"] = weekly["incidence_raw"].clip(upper=cap_value)

    weekly = weekly.set_index("date").sort_index().asfreq("W-MON")

    weekly["departamento"] = region
    weekly["year"] = weekly.index.year

    weekly["incidence_raw"] = weekly["incidence_raw"].fillna(0)

    weekly["incidence_smooth"] = (
        weekly["incidence_raw"]
        .rolling(window=SMOOTH_WINDOW, min_periods=1)
        .mean()
    )

    weekly = weekly.reset_index()

    return weekly[
        ["date", "year", "departamento", "incidence_raw", "incidence_smooth"]
    ]


# TRAINER CLASS

class BaselineStatTrainer:
    def __init__(self, region_df, region_name):
        self.region_df = region_df.copy()
        self.region_name = region_name
        self.results = []

    def load_dataset(self):
        self.region_df["date"] = pd.to_datetime(self.region_df["date"])
        self.region_df = self.region_df.sort_values("date").reset_index(drop=True)

        self.series_original = self.region_df["incidence_smooth"].values.astype(float)
        self.series_model = transform_y(self.series_original)
        self.dates = self.region_df["date"].values

    @staticmethod
    def forecast_naive(train_model, horizon):
        preds_model = np.repeat(train_model[-1], horizon)
        return inverse_transform_y(preds_model)

    @staticmethod
    def forecast_seasonal_naive(train_model, horizon):
        if len(train_model) < SEASONAL_PERIOD:
            preds_model = np.repeat(train_model[-1], horizon)
        else:
            last_season = train_model[-SEASONAL_PERIOD:]
            preds_model = np.array([last_season[i % SEASONAL_PERIOD] for i in range(horizon)])

        return inverse_transform_y(preds_model)

    @staticmethod
    def forecast_holt_winters(train_model, horizon):
        try:
            fit = ExponentialSmoothing(
                train_model,
                trend="add",
                damped_trend=True,
                seasonal="add",
                seasonal_periods=SEASONAL_PERIOD,
                initialization_method="estimated"
            ).fit(optimized=True)

            preds_model = np.array(fit.forecast(horizon))

        except Exception:
            preds_model = np.repeat(train_model[-1], horizon)

        preds = inverse_transform_y(preds_model)

        train_original = inverse_transform_y(train_model)
        upper_clip = np.percentile(train_original, 99)

        return np.clip(preds, 0, upper_clip)

    @staticmethod
    def forecast_arima(train_model, horizon):
        best_fit = None
        best_aic = np.inf

        for order in ARIMA_ORDERS:
            try:
                fit = ARIMA(train_model, order=order).fit()
                if fit.aic < best_aic:
                    best_aic = fit.aic
                    best_fit = fit
            except Exception:
                continue

        if best_fit is None:
            preds_model = np.repeat(train_model[-1], horizon)
        else:
            preds_model = np.array(best_fit.forecast(horizon))

        preds = inverse_transform_y(preds_model)

        train_original = inverse_transform_y(train_model)
        upper_clip = np.percentile(train_original, 99)

        return np.clip(preds, 0, upper_clip)

    def run_model(self, train_model, model_name, horizon):
        if model_name == "naive":
            return self.forecast_naive(train_model, horizon)

        if model_name == "seasonal_naive":
            return self.forecast_seasonal_naive(train_model, horizon)

        if model_name == "holt_winters":
            return self.forecast_holt_winters(train_model, horizon)

        if model_name == "arima":
            return self.forecast_arima(train_model, horizon)

        raise ValueError(f"Unknown model: {model_name}")

    def train(self, model_name):
        start = TRAIN_WINDOW_WEEKS

        while start + FORECAST_HORIZON <= len(self.series_model):
            train_start = start - TRAIN_WINDOW_WEEKS

            train_window_model = self.series_model[train_start:start]

            test_window_original = self.series_original[start:start + FORECAST_HORIZON]
            test_dates = self.dates[start:start + FORECAST_HORIZON]

            preds_original = self.run_model(
                train_model=train_window_model,
                model_name=model_name,
                horizon=FORECAST_HORIZON
            )

            for i in range(FORECAST_HORIZON):
                self.results.append([
                    str(test_dates[i])[:10],
                    test_window_original[i],
                    preds_original[i],
                    model_name,
                    self.region_name
                ])

            start += STEP_WEEKS

    def forecast_future(self, model_name):
        history_original = self.series_original.copy()
        history_model = transform_y(history_original)

        future_rows = []
        last_actual_date = pd.to_datetime(self.region_df["date"].iloc[-1])

        for block in range(FUTURE_BLOCKS):
            train_window_model = history_model[-TRAIN_WINDOW_WEEKS:]

            block_preds_original = self.run_model(
                train_model=train_window_model,
                model_name=model_name,
                horizon=FORECAST_HORIZON
            )

            for j, pred_val in enumerate(block_preds_original, start=1):
                future_date = last_actual_date + pd.Timedelta(
                    weeks=(block * FORECAST_HORIZON) + j
                )

                future_rows.append({
                    "date": future_date,
                    "predicted": pred_val,
                    "model": model_name,
                    "region": self.region_name
                })

            history_original = np.append(history_original, block_preds_original)
            history_model = transform_y(history_original)

        return pd.DataFrame(future_rows)

    def results_dataframe(self):
        return pd.DataFrame(
            self.results,
            columns=["date", "actual", "predicted", "model", "region"]
        )

    
    def comparison_dataframe(self):
        df = self.results_dataframe().copy()

        comparison = (
            df.pivot_table(
                index=["date", "region", "actual"],
                columns="model",
                values="predicted",
                aggfunc="first"
            )
            .reset_index()
        )
        comparison.columns.name = None

        model_columns = [
            c for c in comparison.columns
            if c not in ["date", "region", "actual"]
        ]

        for model in model_columns:
            comparison[f"{model}_error"] = (comparison[model] - comparison["actual"]).round(3)
            comparison[f"{model}_abs_error"] = comparison[f"{model}_error"].abs()
            comparison[f"{model}_pct_error"] = np.where(
                comparison["actual"] != 0,
                (comparison[f"{model}_abs_error"] / comparison["actual"] * 100).round(1),
                np.nan
            )

        abs_error_cols = [f"{m}_abs_error" for m in model_columns]
        if abs_error_cols:
            comparison["best_model"] = (
                comparison[abs_error_cols]
                .idxmin(axis=1)
                .str.replace("_abs_error", "", regex=False)
            )

        return comparison

    def metrics_dataframe(self):
        df = self.results_dataframe().copy()

        rows = []
        for model_name in df["model"].unique():
            temp = df[df["model"] == model_name].copy()

            rows.append({
                "region": self.region_name,
                "model": model_name,
                "mae": round(mean_absolute_error(temp["actual"], temp["predicted"]), 3),
                "rmse": round(safe_rmse(temp["actual"], temp["predicted"]), 3),
                "r2": round(r2_score(temp["actual"], temp["predicted"]), 3)
            })

        return pd.DataFrame(rows)

    
    def future_predictions_dataframe(self, future_df):
        df = future_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        pivot = (
            df.pivot_table(
                index=["date", "region"],
                columns="model",
                values="predicted",
                aggfunc="first"
            )
            .reset_index()
        )
        pivot.columns.name = None
        pivot["date"] = pivot["date"].dt.strftime("%Y-%m-%d")

        return pivot.sort_values("date").reset_index(drop=True)

    def plot_results(self, future_df, output_png):
        df = self.results_dataframe().copy()
        df["date"] = pd.to_datetime(df["date"])

        actual_df = self.region_df[["date", "incidence_smooth"]].copy()
        actual_df["date"] = pd.to_datetime(actual_df["date"])
        actual_df = actual_df.rename(columns={"incidence_smooth": "actual"}).sort_values("date")

        last_actual_date = actual_df["date"].max()
        last_actual_value = actual_df["actual"].iloc[-1]

        future_df = future_df.copy()
        future_df["date"] = pd.to_datetime(future_df["date"])

        fig, (ax_plot, ax_table) = plt.subplots(
            2, 1, figsize=(18, 14), gridspec_kw={"height_ratios": [4, 2.8]}
        )

        ax_plot.plot(
            actual_df["date"],
            actual_df["actual"],
            label="Actual",
            color=MODEL_COLORS["actual"],
            linewidth=2.5
        )

        for model_name in df["model"].unique():
            temp = df[df["model"] == model_name].sort_values("date")

            ax_plot.plot(
                temp["date"],
                temp["predicted"],
                label=model_name,
                color=MODEL_COLORS[model_name],
                linewidth=2
            )

            fut = future_df[future_df["model"] == model_name].sort_values("date").copy()

            if not fut.empty:
                fut_connected = pd.concat([
                    pd.DataFrame({
                        "date": [last_actual_date],
                        "predicted": [last_actual_value],
                        "model": [model_name],
                        "region": [self.region_name]
                    }),
                    fut
                ], ignore_index=True)

                ax_plot.plot(
                    fut_connected["date"],
                    fut_connected["predicted"],
                    label=f"{model_name} - Future",
                    color=MODEL_COLORS[model_name],
                    linestyle="--",
                    linewidth=2
                )

        xlim_end = future_df["date"].max() if not future_df.empty else last_actual_date
        ax_plot.set_xlim(actual_df["date"].min(), xlim_end)

        ax_plot.axvline(last_actual_date, color="gray", linestyle=":", linewidth=1.5)

        if not future_df.empty:
            ax_plot.text(
                last_actual_date,
                ax_plot.get_ylim()[1],
                "Start of \n projection",
                fontsize=8,
                color="black",
                verticalalignment="top"
            )

        ax_plot.set_title(
            f"{self.region_name} | Children 5- Deaths | Statistical Models | "
            f"1-Month Ahead Backtest + 1-Year Future Projection",
            fontsize=14,
            fontweight="bold"
        )

        ax_plot.set_xlabel("Date", fontsize=11)
        ax_plot.set_ylabel("Incidence per 100 000 inhabitants", fontsize=11)
        ax_plot.grid(alpha=0.3)
        ax_plot.legend(fontsize=9, loc="upper left")

        ax_table.axis("off")

        if not future_df.empty:
            table_df = (
                future_df
                .pivot_table(index="date", columns="model", values="predicted", aggfunc="first")
                .reset_index()
            )
            table_df.columns.name = None
            table_df["date"] = pd.to_datetime(table_df["date"]).dt.strftime("%Y-%m-%d")
            table_df = table_df.round(3)

            cols = ["date"]
            for model_key in ["naive", "seasonal_naive", "holt_winters", "arima"]:
                if model_key in table_df.columns:
                    table_df = table_df.rename(columns={model_key: display_name_for(model_key)})
                    cols.append(display_name_for(model_key))

            table_df = table_df[cols]

            ax_table.text(
                0.5, 1.02,
                "Future predictions of deaths in children 5- (per 100,000 inhabitants)",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
                transform=ax_table.transAxes
            )

            table = ax_table.table(
                cellText=table_df.values,
                colLabels=table_df.columns,
                loc="center",
                cellLoc="center",
                colLoc="center",
                bbox=[0.05, 0.00, 0.90, 0.90]
            )

            table.auto_set_font_size(True)
            table.set_fontsize(7)

            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_text_props(weight="bold")
                    cell.set_facecolor("#D9EAD3")
                else:
                    cell.set_facecolor("#FFF9F0")

        else:
            ax_table.text(
                0.5, 0.5, "No future predictions are available.",
                ha="center", va="center", fontsize=11
            )

        plt.subplots_adjust(left=0.07, right=0.97, top=0.94, bottom=0.001, hspace=0.2)
        plt.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.close()

    
    def plot_heatmap(self, future_df, output_png):
        historical = self.region_df.copy()
        historical["date"] = pd.to_datetime(historical["date"])
        historical = historical[["date", "incidence_smooth"]].copy()

        future = future_df.copy()
        future["date"] = pd.to_datetime(future["date"])

        if "model" in future.columns:
            future = future[future["model"] == BEST_MODEL_FOR_PLOTS].copy()

        future = future.rename(columns={"predicted": "incidence_smooth"})
        future = future[["date", "incidence_smooth"]].copy()

        df = pd.concat([historical, future], ignore_index=True)
        df["year"] = df["date"].dt.year
        df["week"] = df["date"].dt.isocalendar().week.astype(int)

        heatmap = (
            df.pivot_table(index="year", columns="week", values="incidence_smooth", aggfunc="mean")
            .sort_index()
        )
        heatmap = heatmap.reindex(columns=range(1, 53))

        fig, ax = plt.subplots(figsize=(18, 8))

        vmin = np.nanpercentile(heatmap.values, 5)
        vmax = np.nanpercentile(heatmap.values, 95)

        img = ax.imshow(
            heatmap, aspect="auto", cmap="YlOrRd", interpolation="nearest",
            origin="upper", vmin=vmin, vmax=vmax
        )

        ax.set_title(
            f"{self.region_name}\nWeekly Deaths Incidence Heatmap (Historical + Forecast)",
            fontsize=15, fontweight="bold"
        )

        ax.set_xlabel("Epidemiological Week", fontsize=12)
        ax.set_ylabel("Year", fontsize=12)

        xticks = np.arange(0, 52, 4)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(i + 1) for i in xticks])

        ax.set_yticks(np.arange(len(heatmap.index)))
        ax.set_yticklabels(heatmap.index.astype(str))

        last_hist_year = historical["date"].dt.year.max()
        forecast_year = last_hist_year + 1 if not future.empty else None

        if forecast_year is not None and forecast_year in heatmap.index:
            idx = list(heatmap.index).index(forecast_year)
            ax.axhline(idx - 0.5, color="black", linestyle="--", linewidth=2)
            ax.text(
                52.5, idx, "Forecast", fontsize=10, fontweight="bold", va="center"
            )

        cbar = plt.colorbar(img, ax=ax)
        cbar.set_label("Incidence per 100,000 inhabitants", fontsize=11)

        plt.tight_layout()
        plt.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_anomalies(self, future_df, output_png):
        historical = self.region_df.copy()
        historical["date"] = pd.to_datetime(historical["date"])
        historical["week"] = historical["date"].dt.isocalendar().week.astype(int)
        historical = historical[["date", "week", "incidence_smooth"]].copy()
        historical["source"] = "historical"

        weekly_stats = (
            historical
            .groupby("week")["incidence_smooth"]
            .agg(["mean", "std"])
            .reset_index()
        )
        weekly_stats = weekly_stats.rename(columns={"mean": "historical_mean", "std": "historical_std"})
        weekly_stats["week"] = weekly_stats["week"].astype(int)
        weekly_stats["historical_std"] = weekly_stats["historical_std"].fillna(0)

        future = future_df.copy()
        future["date"] = pd.to_datetime(future["date"])

        if "model" in future.columns:
            future = future[future["model"] == BEST_MODEL_FOR_PLOTS].copy()

        if future.empty:
            future = pd.DataFrame(columns=["date", "predicted", "model", "region"])

        if not future.empty:
            future["week"] = future["date"].dt.isocalendar().week.astype(int)
            future = future.rename(columns={"predicted": "incidence_smooth"})
            future = future[["date", "week", "incidence_smooth"]].copy()
            future["source"] = "forecast"

        df = pd.concat([historical, future], ignore_index=True)
        df["week"] = df["week"].astype(int)

        df = df.merge(weekly_stats, on="week", how="left")

        df["anomaly"] = df["incidence_smooth"] - df["historical_mean"]

        df["z_score"] = np.where(
            df["historical_std"] > 0,
            df["anomaly"] / df["historical_std"],
            0
        )

        def classify_anomaly(z):
            if z >= 2:
                return "Very high"
            elif z >= 1:
                return "High"
            elif z <= -2:
                return "Very low"
            elif z <= -1:
                return "Low"
            else:
                return "Normal"

        df["classification"] = df["z_score"].apply(classify_anomaly)

        color_map = {
            "Very high": "#8B0000",
            "High": "#FF4D4D",
            "Normal": "#BDBDBD",
            "Low": "#90EE90",
            "Very low": "#006400"
        }

        df["color"] = df["classification"].map(color_map)

        hist_df = df[df["source"] == "historical"].copy()
        fore_df = df[df["source"] == "forecast"].copy()

        fig, ax = plt.subplots(figsize=(18, 8))

        if not hist_df.empty:
            ax.bar(
                hist_df["date"], hist_df["anomaly"], color=hist_df["color"],
                width=5, alpha=0.85, label="Historical"
            )

        if not fore_df.empty:
            ax.bar(
                fore_df["date"], fore_df["anomaly"], color=fore_df["color"],
                width=5, alpha=0.55, edgecolor="black", linewidth=0.5,
                label=f"{display_name_for(BEST_MODEL_FOR_PLOTS)} Projection"
            )

        ax.axhline(0, color="black", linewidth=1.5)

        if not fore_df.empty:
            forecast_start = fore_df["date"].min()
            ax.axvline(forecast_start, color="black", linestyle="--", linewidth=1.5)

            ymax = ax.get_ylim()[1]
            ax.text(
                forecast_start, ymax * 0.92, "Start of \n projection",
                rotation=90, verticalalignment="top", horizontalalignment="right", fontsize=9
            )

        ax.set_title(
            f"{self.region_name} | Epidemiological Anomalies in Deaths of Children 5-",
            fontsize=14, fontweight="bold"
        )

        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel(
            "Deviation from Historical Pattern (incidence per 100 000 inhabitants)", fontsize=10
        )
        ax.grid(axis="y", alpha=0.25)

        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#8B0000", label="Very high alert"),
            Patch(facecolor="#FF4D4D", label="Increase"),
            Patch(facecolor="#BDBDBD", label="Expected Behavior"),
            Patch(facecolor="#90EE90", label="Decrease"),
            Patch(facecolor="#006400", label="Very low alert")
        ]

        ax.legend(
            handles=legend_elements, title="Anomaly Level", loc="upper left",
            fontsize=9, title_fontsize=9, ncol=2
        )

        ax.text(
            0.99, 0.02,
            "The anomaly compares each week with the historical average of the same epidemiological week.",
            transform=ax.transAxes, fontsize=8.5, horizontalalignment="right",
            verticalalignment="bottom", bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8)
        )

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.close()


# MAIN

def main():
    ira, pop = load_data()

    all_results = []
    all_metrics = []
    all_future = []
    all_comparisons = []

    model_names = ["naive", "seasonal_naive", "holt_winters", "arima"]

    for region in REGIONS:
        print(f"\nRunning region: {region}")

        df_region = prepare_region_series(ira, pop, region)
        print(f"{region}: {len(df_region)} filas")

        trainer = BaselineStatTrainer(df_region, region)
        trainer.load_dataset()

        for model_name in model_names:
            print(f"  Running model: {model_name}")
            trainer.train(model_name)

        future_dfs = []
        for model_name in model_names:
            print(f"  Forecasting: {model_name}")
            future_dfs.append(trainer.forecast_future(model_name))

        future_df = pd.concat(future_dfs, ignore_index=True)

        results_df = trainer.results_dataframe()
        metrics_df = trainer.metrics_dataframe()
        comparison_df = trainer.comparison_dataframe()
        future_csv_df = trainer.future_predictions_dataframe(future_df)

        all_results.append(results_df)
        all_metrics.append(metrics_df)
        all_future.append(future_csv_df)
        all_comparisons.append(comparison_df)

        prefix = region.lower()
        region_dir = OUTPUT_DIR / prefix
        region_dir.mkdir(parents=True, exist_ok=True)

        results_df.to_csv(
            region_dir / f"{prefix}_children_death_baseline_stat_predictions.csv",
            index=False
        )

        metrics_df.to_csv(
            region_dir / f"{prefix}_children_death_baseline_stat_metrics.csv",
            index=False
        )

        future_csv_df.to_csv(
            region_dir / f"{prefix}_children_death_baseline_stat_future_predictions.csv",
            index=False
        )

        comparison_df.to_csv(
            region_dir / f"{prefix}_children_death_baseline_stat_comparison_predictions.csv",
            index=False
        )

        trainer.plot_results(
            future_df=future_df,
            output_png=region_dir / f"{prefix}_children_death_baseline_stat_plot.png"
        )

        trainer.plot_heatmap(
            future_df=future_df,
            output_png=region_dir / f"{prefix}_children_death_heatmap.png"
        )

        trainer.plot_anomalies(
            future_df=future_df,
            output_png=region_dir / f"{prefix}_children_death_anomalies.png"
        )

    pd.concat(all_results, ignore_index=True).to_csv(
        OUTPUT_DIR / "all_regions_children_death_baseline_stat_predictions.csv",
        index=False
    )

    pd.concat(all_metrics, ignore_index=True).to_csv(
        OUTPUT_DIR / "all_regions_children_death_baseline_stat_metrics.csv",
        index=False
    )

    pd.concat(all_future, ignore_index=True).to_csv(
        OUTPUT_DIR / "all_regions_children_death_baseline_stat_future_predictions.csv",
        index=False
    )

    pd.concat(all_comparisons, ignore_index=True).to_csv(
        OUTPUT_DIR / "all_regions_children_death_baseline_stat_comparison_predictions.csv",
        index=False
    )

    print(f"\nDone. Outputs saved to: {OUTPUT_DIR}")

    fin = time.perf_counter()
    tiempo = fin - inicio
    print(f"Tiempo de ejecución: {tiempo:.4f} segundos")


if __name__ == "__main__":
    main()