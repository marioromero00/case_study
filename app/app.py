from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data_prep import (
    prepare_data,
    append_incremental
)

from src.models import MODELS

from src.backtest import (
    run_backtest,
    summarize_backtest,
    select_best_model
)

from src.forecasting import (
    generate_forecast_all,
    generate_prophet_band_all,
    generate_ensemble_band_all,
    save_forecast
)

from src.utils import (
    plot_single_forecast,
    business_summary
)

st.set_page_config(
    page_title="Pricing Forecast Intelligence",
    layout="wide"
)

st.title("🐟 Salmon Pricing Forecast")

st.write(
    """
    Flujo:

    Histórico → Semana nueva → Reentrenar → Seleccionar mejor modelo → Forecast
    """
)

HORIZON = 78

BASE_FILE = ROOT / "data" / "raw" / "Dataset_case_study.csv"

uploaded_file = st.file_uploader(
    "Sube únicamente la nueva semana",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Sube el archivo incremental semanal.")
    st.stop()

# ---------------------------
# Guardar incremental
# ---------------------------

incremental_dir = ROOT / "data" / "incremental"
incremental_dir.mkdir(parents=True, exist_ok=True)

incremental_path = incremental_dir / uploaded_file.name

with open(incremental_path, "wb") as f:
    f.write(uploaded_file.getbuffer())

# ---------------------------
# Cargar histórico
# ---------------------------

with st.spinner("Leyendo histórico..."):

    _, wide_hist = prepare_data(
        BASE_FILE,
        output_dir=ROOT / "data" / "processed"
    )

# ---------------------------
# Cargar incremental
# ---------------------------

with st.spinner("Leyendo nueva semana..."):

    _, wide_new = prepare_data(
        incremental_path,
        output_dir=ROOT / "data" / "processed"
    )

# ---------------------------
# Merge histórico + nueva semana
# ---------------------------

wide_updated = append_incremental(
    wide_hist,
    wide_new
)

st.success(
    f"Histórico actualizado: {wide_updated.shape[0]} semanas"
)

# ---------------------------
# Entrenar
# ---------------------------

if st.button("🚀 Reentrenar y Forecast"):

    with st.spinner("Backtesting modelos..."):

        bt = run_backtest(
            wide_updated,
            MODELS,
            horizon=HORIZON,
            n_folds=4,
            step=13
        )

        summary = summarize_backtest(bt)

        best_model, ranking = select_best_model(
            bt,
            tramo="corto 1-26"
        )

    st.success(
        f"Mejor modelo seleccionado: {best_model}"
    )

    st.subheader("Ranking modelos")

    st.dataframe(
        ranking,
        use_container_width=True
    )

    st.subheader("Backtest")

    st.dataframe(
        summary,
        use_container_width=True
    )

    # -----------------------
    # Forecast
    # -----------------------

    with st.spinner("Generando forecast..."):

        if best_model == "Prophet":

            forecast_df = generate_prophet_band_all(
                wide_updated,
                horizon=HORIZON
            )

        elif best_model == "Ensemble Naive+Prophet":

            forecast_df = generate_ensemble_band_all(
                wide_updated,
                horizon=HORIZON
            )

        else:

            forecast_df = generate_forecast_all(
                wide_updated,
                best_model,
                horizon=HORIZON
            )

        save_forecast(
            forecast_df,
            ROOT / "data" / "forecasts" / "forecast_latest.csv"
        )

    # -----------------------
    # Visualización
    # -----------------------

    selected_serie = st.selectbox(
        "Serie",
        list(wide_updated.columns)
    )

    serie_hist = wide_updated[selected_serie].dropna()

    serie_fc = (
        forecast_df[
            forecast_df["serie"] == selected_serie
        ]
        .sort_values("fecha")
    )

    fig = plot_single_forecast(
        serie_hist,
        serie_fc,
        title=selected_serie
    )

    st.pyplot(fig)

    st.subheader("Resumen Ejecutivo")

    st.dataframe(
        pd.DataFrame(
            business_summary(
                wide_updated,
                forecast_df
            )
        ),
        use_container_width=True
    )

    st.subheader("Forecast completo")

    st.dataframe(
        forecast_df,
        use_container_width=True
    )

    st.download_button(
        "📥 Descargar Forecast",
        forecast_df.to_csv(index=False),
        file_name="forecast_18_meses.csv",
        mime="text/csv"
    )