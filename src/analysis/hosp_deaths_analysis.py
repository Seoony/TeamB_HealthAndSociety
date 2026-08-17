from pathlib import Path
 
import numpy as np
import pandas as pd
import matplotlib
 
matplotlib.use("Agg")
 
import matplotlib.pyplot as plt
 
from scipy.stats import pearsonr, spearmanr
 
 
# ==========================================================
# CONFIGURACIÓN
# ==========================================================
 
ROOT_DIR = Path(__file__).resolve().parents[2]
 
IRA_PATH = ROOT_DIR / "data" / "raw" / "iras_data_raw_temp.parquet"
POP_PATH = ROOT_DIR / "data" / "raw" / "population_dept_long.parquet"
 
OUTPUT_DIR = ROOT_DIR / "outputs" / "hosp_deaths_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
 
REGIONS = ["AREQUIPA", "TACNA", "MOQUEGUA", "TUMBES", "LIMA"]
 
START_YEAR = 2006
END_YEAR = 2023
 
HOSP_COL = "hospitalizados_60mas"
DEATH_COL = "defunciones_60mas"
 
MAX_LAG = 8
 
 
# ==========================================================
# CARGA DE DATOS
# ==========================================================
 
def load_data():

    ira = pd.read_parquet(IRA_PATH)
 
    ira.columns = (
        ira.columns
        .str.lower()
        .str.strip()
    )
 
    required_cols = [
        "iddpto",
        "departamento",
        "ano",
        "semana",
        HOSP_COL,
        DEATH_COL
    ]
 
    missing = [
        col for col in required_cols
        if col not in ira.columns
    ]
 
    if missing:
        raise KeyError(
            f"Faltan columnas requeridas: {missing}"
        )
 
    ira["iddpto"] = (
        ira["iddpto"]
        .astype(str)
        .str.zfill(2)
    )
 
    ira["departamento"] = (
        ira["departamento"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
 
    ira["ano"] = pd.to_numeric(
        ira["ano"],
        errors="coerce"
    )
 
    ira["semana"] = pd.to_numeric(
        ira["semana"],
        errors="coerce"
    )
 
    ira[HOSP_COL] = pd.to_numeric(
        ira[HOSP_COL],
        errors="coerce"
    ).fillna(0)
 
    ira[DEATH_COL] = pd.to_numeric(
        ira[DEATH_COL],
        errors="coerce"
    ).fillna(0)
 
    # Fecha epidemiológica
    ira["date"] = pd.to_datetime(
        ira["ano"].astype("Int64").astype(str)
        + "-"
        + ira["semana"].astype("Int64").astype(str).str.zfill(2)
        + "-1",
        format="%G-%V-%u",
        errors="coerce"
    )
 
    ira = ira.dropna(
        subset=["date"]
    ).copy()
 
    ira["year"] = ira["date"].dt.year
 
    ira = ira[
        (ira["year"] >= START_YEAR)
        &
        (ira["year"] <= END_YEAR)
    ].copy()
 
    return ira
 
 
# ==========================================================
# PREPARAR SERIE REGIONAL
# ==========================================================
 
def prepare_region_series(ira, region_name):
 
    region = region_name.upper()
 
    df = ira[
        ira["departamento"] == region
    ].copy()
 
    if df.empty:
        raise ValueError(
            f"No existen datos para {region}"
        )
 
    weekly = (
        df.groupby(
            ["date", "year", "iddpto", "departamento"],
            as_index=False
        )[
            [HOSP_COL, DEATH_COL]
        ]
        .sum()
    )
 
    weekly = weekly.sort_values("date")
 
    # Crear frecuencia semanal completa
    weekly = (
        weekly
        .set_index("date")
        .asfreq("W-MON")
    )
 
    weekly["departamento"] = region
 
    weekly["year"] = weekly.index.year
 
    weekly[HOSP_COL] = (
        weekly[HOSP_COL]
        .fillna(0)
    )
 
    weekly[DEATH_COL] = (
        weekly[DEATH_COL]
        .fillna(0)
    )
 
    weekly = weekly.reset_index()
 
    return weekly[
        [
            "date",
            "year",
            "departamento",
            HOSP_COL,
            DEATH_COL
        ]
    ]
 
 
# ==========================================================
# CALCULAR LETALIDAD
# ==========================================================
 
def calculate_case_fatality(df):
 
    df = df.copy()
 
    df["fatality_rate"] = np.where(
        df[HOSP_COL] > 0,
        (
            df[DEATH_COL]
            /
            df[HOSP_COL]
        ) * 100,
        np.nan
    )
 
    return df
 
 
# ==========================================================
# CORRELACIÓN POR LAG
# ==========================================================
 
def calculate_lag_correlations(df):
 
    results = []
 
    for lag in range(MAX_LAG + 1):
 
        temp = df[
            [
                HOSP_COL,
                DEATH_COL
            ]
        ].copy()
 
        temp["hospitalizations"] = temp[HOSP_COL]

        temp["deaths"] = (
            temp[DEATH_COL]
            .shift(-lag)
        )
 
        temp = temp[
            [
                "hospitalizations",
                "deaths"
            ]
        ].dropna()
 
        if len(temp) < 3:
            continue
 
        x = temp["hospitalizations"]
        y = temp["deaths"]
 
        if x.nunique() <= 1 or y.nunique() <= 1:
            continue
 
        pearson = pearsonr(x, y)[0]
        spearman = spearmanr(x, y).statistic
 
        results.append({
            "lag_weeks": lag,
            "pearson": pearson,
            "spearman": spearman
        })
 
    return pd.DataFrame(results)
 
 
# ==========================================================
# GRÁFICO DE SERIES TEMPORALES
# ==========================================================
 
def plot_time_series(df, region_name, output_png):
 
    fig, ax1 = plt.subplots(
        figsize=(18, 8)
    )
 
    ax2 = ax1.twinx()
 
    # ==========================================================
    # HOSPITALIZACIONES
    # ==========================================================
    line1, = ax1.plot(
        df["date"],
        df[HOSP_COL],
        color="royalblue",
        linewidth=1.8,
        label="Hospitalizaciones 60+"
    )
 
    # ==========================================================
    # DEFUNCIONES
    # ==========================================================
    line2, = ax2.plot(
        df["date"],
        df[DEATH_COL],
        color="crimson",
        linewidth=1.8,
        label="Defunciones 60+"
    )
 
    # ==========================================================
    # TÍTULO
    # ==========================================================
    ax1.set_title(
        f"{region_name} | Hospitalizaciones y defunciones en adultos de 60+",
        fontsize=14,
        fontweight="bold"
    )
 
    # ==========================================================
    # ETIQUETAS DE EJES
    # ==========================================================
    ax1.set_xlabel("Fecha")
 
    ax1.set_ylabel(
        "Hospitalizaciones",
        color="royalblue",
        fontweight="bold"
    )
 
    ax2.set_ylabel(
        "Defunciones",
        color="crimson",
        fontweight="bold"
    )
 
    # ==========================================================
    # COLOR DE LOS VALORES DE LOS EJES
    # ==========================================================
    ax1.tick_params(
        axis="y",
        labelcolor="royalblue"
    )
 
    ax2.tick_params(
        axis="y",
        labelcolor="crimson"
    )
 
    # ==========================================================
    # CUADRÍCULA
    # ==========================================================
    ax1.grid(
        alpha=0.3
    )
 
    # ==========================================================
    # LEYENDA
    # ==========================================================
    ax1.legend(
        [line1, line2],
        ["Hospitalizaciones 60+", "Defunciones 60+"],
        loc="upper left",
        frameon=True
    )
 
    # ==========================================================
    # AJUSTE Y GUARDADO
    # ==========================================================
    plt.tight_layout()
 
    plt.savefig(
        output_png,
        dpi=300,
        bbox_inches="tight"
    )
 
    plt.close()
 
 
# ==========================================================
# GRÁFICO HOSPITALIZACIONES VS DEFUNCIONES
# ==========================================================
 
def plot_scatter(df, region_name, output_png):
 
    x = df[HOSP_COL]
    y = df[DEATH_COL]
 
    mask = (
        x.notna()
        &
        y.notna()
    )
 
    x = x[mask]
    y = y[mask]
 
    fig, ax = plt.subplots(
        figsize=(10, 8)
    )
 
    # ==========================================================
    # PUNTOS: HOSPITALIZACIONES VS DEFUNCIONES
    # ==========================================================
    ax.scatter(
        x,
        y,
        alpha=0.55,
        color="steelblue",
        edgecolors="black",
        linewidths=0.5,
        label="Observaciones semanales"
    )
 
    # ==========================================================
    # TENDENCIA LINEAL
    # ==========================================================
    if len(x) >= 2:
 
        coef = np.polyfit(
            x,
            y,
            1
        )
 
        trend = np.poly1d(coef)
 
        x_line = np.linspace(
            x.min(),
            x.max(),
            100
        )
 
        ax.plot(
            x_line,
            trend(x_line),
            color="darkred",
            linestyle="--",
            linewidth=2,
            label="Tendencia lineal"
        )
 
        # ======================================================
        # CORRELACIÓN DE PEARSON
        # ======================================================
        r = pearsonr(x, y)[0]
 
        ax.text(
            0.05,
            0.88,
            f"Correlación de Pearson: r = {r:.3f}",
            transform=ax.transAxes,
            fontsize=10
        )
 
    # ==========================================================
    # TÍTULOS
    # ==========================================================
    ax.set_title(
        f"{region_name} | Relación entre hospitalizaciones y defunciones",
        fontsize=13,
        fontweight="bold"
    )
 
    ax.set_xlabel(
        "Hospitalizaciones 60+"
    )
 
    ax.set_ylabel(
        "Defunciones 60+"
    )
 
    # ==========================================================
    # CUADRÍCULA
    # ==========================================================
    ax.grid(
        alpha=0.3
    )
 
    # ==========================================================
    # LEYENDA
    # ==========================================================
    ax.legend(
        loc="upper left"
    )
 
    plt.tight_layout()
 
    plt.savefig(
        output_png,
        dpi=300,
        bbox_inches="tight"
    )
 
    plt.close()
 
 
# ==========================================================
# GRÁFICO DE CORRELACIÓN POR LAG
# ==========================================================
 
def plot_lag_correlation(
    lag_df,
    region_name,
    output_png
):
 
    fig, ax = plt.subplots(
        figsize=(12, 7)
    )
 
    ax.plot(
        lag_df["lag_weeks"],
        lag_df["pearson"],
        marker="o",
        linewidth=2,
        label="Pearson"
    )
 
    ax.plot(
        lag_df["lag_weeks"],
        lag_df["spearman"],
        marker="s",
        linewidth=2,
        label="Spearman"
    )
 
    ax.axhline(
        0,
        color="black",
        linewidth=1
    )
 
    ax.set_title(
        f"{region_name} | Correlación entre hospitalizaciones y defunciones según desfase",
        fontsize=13,
        fontweight="bold"
    )
 
    ax.set_xlabel(
        "Desfase temporal (semanas)"
    )
 
    ax.set_ylabel(
        "Coeficiente de correlación"
    )
 
    ax.set_xticks(
        lag_df["lag_weeks"]
    )
 
    ax.grid(
        alpha=0.3
    )
 
    ax.legend()
 
    plt.tight_layout()
 
    plt.savefig(
        output_png,
        dpi=300,
        bbox_inches="tight"
    )
 
    plt.close()
 
 
def create_annual_summary_table(df, region_name, output_csv, output_png):
 
    # ==========================================================
    # PREPARAR DATOS
    # ==========================================================
 
    data = df.copy()
 
    data["date"] = pd.to_datetime(data["date"])
 
    data["year"] = data["date"].dt.year
 
    data[HOSP_COL] = pd.to_numeric(
        data[HOSP_COL],
        errors="coerce"
    ).fillna(0)
 
    data[DEATH_COL] = pd.to_numeric(
        data[DEATH_COL],
        errors="coerce"
    ).fillna(0)
 
    # ==========================================================
    # RESUMEN ANUAL
    # ==========================================================
 
    annual = (
        data
        .groupby("year")
        .agg(
            hospitalizations=(HOSP_COL, "sum"),
            deaths=(DEATH_COL, "sum"),
            weeks=("date", "count")
        )
        .reset_index()
    )
 
    # ==========================================================
    # PROMEDIOS SEMANALES
    # ==========================================================
 
    annual["avg_hospitalizations_week"] = (
        annual["hospitalizations"] /
        annual["weeks"]
    )
 
    annual["avg_deaths_week"] = (
        annual["deaths"] /
        annual["weeks"]
    )
 
    # ==========================================================
    # PROPORCIÓN DEFUNCIONES / HOSPITALIZACIONES
    # ==========================================================
 
    annual["death_hospitalization_ratio"] = np.where(
        annual["hospitalizations"] > 0,
        (
            annual["deaths"] /
            annual["hospitalizations"]
        ) * 100,
        np.nan
    )
 
    # ==========================================================
    # VARIACIÓN INTERANUAL
    # ==========================================================
 
    annual["hospitalization_change_pct"] = (
        annual["hospitalizations"]
        .pct_change()
        .mul(100)
    )
 
    annual["death_change_pct"] = (
        annual["deaths"]
        .pct_change()
        .mul(100)
    )
 
    # ==========================================================
    # REDONDEAR
    # ==========================================================
 
    annual["avg_hospitalizations_week"] = (
        annual["avg_hospitalizations_week"].round(2)
    )
 
    annual["avg_deaths_week"] = (
        annual["avg_deaths_week"].round(2)
    )
 
    annual["death_hospitalization_ratio"] = (
        annual["death_hospitalization_ratio"].round(2)
    )
 
    annual["hospitalization_change_pct"] = (
        annual["hospitalization_change_pct"].round(2)
    )
 
    annual["death_change_pct"] = (
        annual["death_change_pct"].round(2)
    )
 
    # ==========================================================
    # GUARDAR CSV
    # ==========================================================
 
    annual.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig"
    )
 
    # ==========================================================
    # TABLA PARA IMAGEN
    # ==========================================================
 
    table_df = annual.copy()
 
    table_df = table_df.rename(
        columns={
            "year": "Year",
            "hospitalizations": "Hospitalizations",
            "deaths": "Deaths",
            "weeks": "Weeks",
            "avg_hospitalizations_week": "Hosp./week",
            "avg_deaths_week": "Deaths/week",
            "death_hospitalization_ratio": "Death./Hosp. (%)",
            "hospitalization_change_pct": "Var. Hosp. (%)",
            "death_change_pct": "Var. Death. (%)"
        }
    )
 
    # ==========================================================
    # FORMATO VISUAL
    # ==========================================================
 
    table_df["Hospitalizations"] = (
        table_df["Hospitalizations"]
        .astype(int)
    )
 
    table_df["Deaths"] = (
        table_df["Deaths"]
        .astype(int)
    )
 
    table_df["Weeks"] = (
        table_df["Weeks"]
        .astype(int)
    )
 
    # Mostrar "-" para el primer año,
    # ya que no existe año anterior para comparar.
    table_df = table_df.fillna("-")
 
    # ==========================================================
    # FIGURA
    # ==========================================================
 
    fig, ax = plt.subplots(
        figsize=(18, 5 + len(table_df) * 0.25)
    )
 
    ax.axis("off")
 
    ax.set_title(
        f"{region_name} | Annual summary of hospitalizations "
        f"and deaths in adults aged 60 and over",
        fontsize=14,
        fontweight="bold",
        pad=20
    )
 
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center"
    )
 
    # ==========================================================
    # CONFIGURACIÓN DE TABLA
    # ==========================================================
 
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
 
    table.scale(
        1,
        1.8
    )
 
    # Encabezado
    for col in range(len(table_df.columns)):
        cell = table[(0, col)]
 
        cell.set_text_props(
            weight="bold"
        )
 
        cell.set_facecolor(
            "#D9EAD3"
        )
 
    # Cuerpo
    for row in range(1, len(table_df) + 1):
 
        for col in range(len(table_df.columns)):
 
            cell = table[(row, col)]
 
            cell.set_facecolor(
                "#F8F9FA" if row % 2 == 0 else "white"
            )
 
    # ==========================================================
    # GUARDAR PNG
    # ==========================================================
 
    plt.savefig(
        output_png,
        dpi=300,
        bbox_inches="tight"
    )
 
    plt.close()
 
 
# ==========================================================
# MAIN
# ==========================================================
 
def main():
 
    ira = load_data()
 
    print("\n==========================================")
    print("ANÁLISIS HOSPITALIZACIONES - DEFUNCIONES")
    print("==========================================")
 
    for region in REGIONS:
 
        print(
            f"\nAnalizando región: {region}"
        )
 
        region_df = prepare_region_series(
            ira,
            region
        )
 
        region_df = calculate_case_fatality(
            region_df
        )
 
        # --------------------------------------------------
        # Carpeta individual de la región
        # --------------------------------------------------
 
        region_output = (
            OUTPUT_DIR
            / region.lower()
        )
 
        region_output.mkdir(
            parents=True,
            exist_ok=True
        )
 
        # --------------------------------------------------
        # Correlaciones
        # --------------------------------------------------
 
        lag_df = calculate_lag_correlations(
            region_df
        )
 
        # --------------------------------------------------
        # Guardar datos
        # --------------------------------------------------
 
        region_df.to_csv(
            region_output
            / f"{region.lower()}_hosp_deaths_weekly.csv",
            index=False
        )
 
        lag_df.to_csv(
            region_output
            / f"{region.lower()}_lag_correlations.csv",
            index=False
        )
 
 
        # --------------------------------------------------
        # Resumen anual
        # --------------------------------------------------
        create_annual_summary_table(
            region_df,
            region,
            region_output
            / f"{region.lower()}_annual_summary.csv",
            region_output
            / f"{region.lower()}_annual_summary.png"
        )
 
        
        # --------------------------------------------------
        # Gráficos
        # --------------------------------------------------
 
        plot_time_series(
            region_df,
            region,
            region_output
            / f"{region.lower()}_hosp_deaths_timeseries.png"
        )
 
        plot_scatter(
            region_df,
            region,
            region_output
            / f"{region.lower()}_hosp_deaths_scatter.png"
        )
 
        if not lag_df.empty:
 
            plot_lag_correlation(
                lag_df,
                region,
                region_output
                / f"{region.lower()}_lag_correlation.png"
            )
 
        print(
            f"Resultados guardados en: {region_output}"
        )
 
    print(
        "\nAnálisis completado."
    )
 
 
if __name__ == "__main__":
    main()