from pathlib import Path
import pandas as pd

from src.models import MODELS, m_prophet_band


def generate_forecast_for_series(series, model_name, horizon=78):
    model_func = MODELS[model_name]
    pred = model_func(series, horizon)

    future_dates = pd.date_range(
        start=series.index.max() + pd.Timedelta(weeks=1),
        periods=horizon,
        freq="W-MON",
    )

    return pd.DataFrame({
        "fecha": future_dates,
        "forecast": pred,
    })


def generate_forecast_all(wide, model_name, horizon=78):
    outputs = []

    for col in wide.columns:
        s = wide[col].dropna()
        fc = generate_forecast_for_series(s, model_name, horizon=horizon)
        fc["serie"] = col
        fc["modelo"] = model_name

        producto, calibre = col.split("|")
        fc["producto"] = producto
        fc["calibre"] = calibre
        fc["unidad"] = "USD/kg" if producto == "Brazil HON" else "USD/lb"

        outputs.append(fc)

    return pd.concat(outputs, ignore_index=True)


def generate_prophet_band_all(wide, horizon=78, interval=0.80):
    outputs = []

    for col in wide.columns:
        s = wide[col].dropna()
        fc = m_prophet_band(s, horizon=horizon, interval=interval)

        fc = fc.rename(columns={
            "ds": "fecha",
            "yhat_lower": "forecast_lower",
            "yhat": "forecast",
            "yhat_upper": "forecast_upper",
        })

        fc["serie"] = col
        fc["modelo"] = "Prophet"

        producto, calibre = col.split("|")
        fc["producto"] = producto
        fc["calibre"] = calibre
        fc["unidad"] = "USD/kg" if producto == "Brazil HON" else "USD/lb"

        outputs.append(fc)

    return pd.concat(outputs, ignore_index=True)


def save_forecast(forecast_df, output_path="data/forecasts/forecast_latest.csv"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    forecast_df.to_csv(output_path, index=False)

    return str(output_path)