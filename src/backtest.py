import numpy as np
import pandas as pd


def safe_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    den = np.where(np.abs(y_true) < 1e-8, np.nan, np.abs(y_true))

    return np.nanmean(np.abs(y_pred - y_true) / den * 100)


def backtest_series(series, models, horizon=78, n_folds=4, step=13):
    res = []

    min_train = len(series) - horizon - (n_folds - 1) * step

    if min_train <= 52:
        raise ValueError("La serie tiene poca historia para backtest robusto.")

    for k in range(n_folds):
        cut = min_train + k * step

        train = series.iloc[:cut]
        test = series.iloc[cut:cut + horizon]

        if len(test) < 26:
            continue

        h = len(test)

        for name, model_func in models.items():
            try:
                pred = model_func(train, h)
            except Exception:
                pred = np.full(h, np.nan)

            ape = np.abs(pred - test.values) / np.where(
                np.abs(test.values) < 1e-8,
                np.nan,
                np.abs(test.values),
            ) * 100

            for lo, hi, tramo in [
                (0, 26, "corto 1-26"),
                (26, 52, "medio 27-52"),
                (52, 78, "largo 53-78"),
            ]:
                seg = ape[lo:min(hi, h)]

                if len(seg):
                    res.append({
                        "serie": series.name,
                        "fold": k,
                        "modelo": name,
                        "tramo": tramo,
                        "MAPE_%": np.nanmean(seg),
                    })

    return pd.DataFrame(res)


def run_backtest(wide, models, horizon=78, n_folds=4, step=13):
    all_results = []

    for col in wide.columns:
        s = wide[col].dropna()
        s.name = col

        bt = backtest_series(
            s,
            models,
            horizon=horizon,
            n_folds=n_folds,
            step=step,
        )

        all_results.append(bt)

    return pd.concat(all_results, ignore_index=True)


def summarize_backtest(bt):
    return (
        bt.groupby(["modelo", "tramo"], as_index=False)["MAPE_%"]
        .mean()
        .sort_values(["tramo", "MAPE_%"])
    )


def select_best_model(bt, tramo="corto 1-26"):
    aux = bt[bt["tramo"] == tramo]

    ranking = (
        aux.groupby("modelo", as_index=False)["MAPE_%"]
        .mean()
        .sort_values("MAPE_%")
    )

    return ranking.iloc[0]["modelo"], ranking