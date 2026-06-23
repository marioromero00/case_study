import numpy as np
import pandas as pd

from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX


def m_naive(train, horizon, period=52):
    values = train.values
    out = []

    for h in range(1, horizon + 1):
        if len(values) >= period:
            out.append(values[-period + ((h - 1) % period)])
        else:
            out.append(values[-1])

    return np.array(out)


def fit_prophet(train, interval=0.80):
    df = pd.DataFrame({
        "ds": train.index,
        "y": train.values
    })

    lo = train.min()
    hi = train.max()
    margin = (hi - lo) * 0.10

    df["floor"] = lo - margin
    df["cap"] = hi + margin

    model = Prophet(
        growth="logistic",
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=interval,
        changepoint_prior_scale=0.02,
        changepoint_range=0.85,
    )

    model.fit(df)
    model._floor = df["floor"].iloc[0]
    model._cap = df["cap"].iloc[0]

    return model


def m_prophet(train, horizon):
    model = fit_prophet(train)

    future = model.make_future_dataframe(
        periods=horizon,
        freq="W-MON"
    )

    future["floor"] = model._floor
    future["cap"] = model._cap

    pred = model.predict(future)

    return pred["yhat"].values[-horizon:]


def m_prophet_band(train, horizon, interval=0.80):
    model = fit_prophet(train, interval=interval)

    future = model.make_future_dataframe(
        periods=horizon,
        freq="W-MON"
    )

    future["floor"] = model._floor
    future["cap"] = model._cap

    fc = model.predict(future).iloc[-horizon:]

    return fc[["ds", "yhat_lower", "yhat", "yhat_upper"]]


def m_ensemble(train, horizon):
    return (m_naive(train, horizon) + m_prophet(train, horizon)) / 2


def m_ensemble_band(train, horizon, interval=0.80):
    naive_fc = m_naive(train, horizon)
    model = fit_prophet(train, interval=interval)

    future = model.make_future_dataframe(periods=horizon, freq="W-MON")
    future["floor"] = model._floor
    future["cap"] = model._cap

    fc = model.predict(future).iloc[-horizon:].reset_index(drop=True)
    fc["yhat"] = (naive_fc + fc["yhat"].values) / 2

    return fc[["ds", "yhat_lower", "yhat", "yhat_upper"]]


def m_sarima(train, horizon):
    try:
        model = SARIMAX(
            train.values,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 26),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        res = model.fit(disp=False)
        return res.forecast(steps=horizon)

    except Exception:
        return np.full(horizon, np.nan)


MODELS = {
    "Naive estacional": m_naive,
    "Prophet": m_prophet,
    "Ensemble Naive+Prophet": m_ensemble,
    "SARIMA": m_sarima,
}