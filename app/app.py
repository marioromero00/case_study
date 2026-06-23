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
    business_summary,
    price_variation_summary,
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

# ---------------------------
# Resumen del archivo recibido
# ---------------------------

var_df = price_variation_summary(wide_updated)

last_date = wide_updated.index.max()
iso_year, iso_week, _ = last_date.isocalendar()

st.success(
    f"**Archivo recibido** · Semana {iso_year}-{iso_week:02d} · "
    f"{len(wide_updated.columns)} series actualizadas · "
    f"Última actualización: {last_date.strftime('%-d-%b-%Y')}"
)

card_cols = st.columns(3)
for i, col_name in enumerate(wide_updated.columns):
    s = wide_updated[col_name].dropna()
    precio_actual = s.iloc[-1]
    label = col_name.replace("|", " ").replace(" lb", "")

    if len(s) >= 2 and pd.notna(s.iloc[-2]):
        delta_pct = (precio_actual / s.iloc[-2] - 1) * 100
        delta_str = f"{delta_pct:+.1f}%"
    else:
        delta_str = None

    with card_cols[i % 3]:
        st.metric(label=label, value=f"{precio_actual:.2f}", delta=delta_str)

st.divider()

# ---------------------------
# Entrenar
# ---------------------------

if st.button("🚀 Reentrenar y Forecast"):

    with st.spinner("Backtesting modelos..."):
        bt = run_backtest(wide_updated, MODELS, horizon=HORIZON, n_folds=4, step=13)
        summary = summarize_backtest(bt)
        best_model, ranking = select_best_model(bt, tramo="corto 1-26")

    best_mape_corto = ranking.iloc[0]["MAPE_%"]
    best_mape_global = bt[bt["modelo"] == best_model]["MAPE_%"].mean()

    # -----------------------
    # Modelo elegido + MAPEs
    # -----------------------

    c1, c2, c3 = st.columns(3)
    c1.metric("Modelo elegido", best_model)
    c2.metric("MAPE corto (1-26 sem)", f"{best_mape_corto:.1f}%")
    c3.metric("MAPE global", f"{best_mape_global:.1f}%")

    # -----------------------
    # Forecast
    # -----------------------

    with st.spinner("Generando forecast..."):

        if best_model == "Prophet":
            forecast_df = generate_prophet_band_all(wide_updated, horizon=HORIZON)
        elif best_model == "Ensemble Naive+Prophet":
            forecast_df = generate_ensemble_band_all(wide_updated, horizon=HORIZON)
        else:
            forecast_df = generate_forecast_all(wide_updated, best_model, horizon=HORIZON)

        save_forecast(forecast_df, ROOT / "data" / "forecasts" / "forecast_latest.csv")

    # -----------------------
    # Tabla: variación + forecast
    # -----------------------

    st.subheader("Variación histórica y forecast")

    biz = pd.DataFrame(business_summary(wide_updated, forecast_df)).set_index("serie")
    combined = var_df.join(biz[["forecast_6m", "forecast_18m", "var_18m_%"]])
    pct_cols = [c for c in combined.columns if c.startswith("Δ")] + ["var_18m_%"]

    def _color_pct(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return ""
        return "color: #2a9d8f" if val > 0 else "color: #c1502e" if val < 0 else ""

    st.dataframe(
        combined.style
            .format("{:.2f}", subset=["precio_actual", "forecast_6m", "forecast_18m"])
            .format(lambda v: f"{v:+.1f}%" if pd.notna(v) else "—", subset=pct_cols)
            .map(_color_pct, subset=pct_cols),
        use_container_width=True,
    )

    # -----------------------
    # Gráfico por serie
    # -----------------------

    st.subheader("Forecast por serie")

    selected_serie = st.selectbox("Serie", list(wide_updated.columns))

    serie_hist = wide_updated[selected_serie].dropna()
    serie_fc = forecast_df[forecast_df["serie"] == selected_serie].sort_values("fecha")

    fig = plot_single_forecast(serie_hist, serie_fc, title=selected_serie)
    st.pyplot(fig)

    # -----------------------
    # Detalle backtest
    # -----------------------

    with st.expander("Detalle backtest"):
        st.dataframe(
            ranking.rename(columns={"MAPE_%": "MAPE% (corto 1-26)"}),
            use_container_width=True,
        )
        st.dataframe(summary, use_container_width=True)

    # -----------------------
    # Descarga
    # -----------------------

    st.download_button(
        "📥 Descargar Forecast",
        forecast_df.to_csv(index=False),
        file_name="forecast_18_meses.csv",
        mime="text/csv",
    )