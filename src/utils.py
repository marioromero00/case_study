import matplotlib.pyplot as plt
import plotly.graph_objects as go


COLORS = {
    "Brazil HON|10-12 lb": "#1b4965",
    "Brazil HON|12-14 lb": "#2a6f97",
    "Brazil HON|14-16 lb": "#468faf",
    "USA Fillet|2/3": "#9e2a2b",
    "USA Fillet|3/4": "#c1502e",
    "USA Fillet|4/5": "#e09f3e",
}


def plot_single_forecast(series, forecast_df, title=None):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(series.index, series.values, label="Histórico", linewidth=1.5)
    ax.plot(forecast_df["fecha"], forecast_df["forecast"], label="Forecast", linewidth=2)

    if "forecast_lower" in forecast_df.columns and "forecast_upper" in forecast_df.columns:
        ax.fill_between(
            forecast_df["fecha"],
            forecast_df["forecast_lower"],
            forecast_df["forecast_upper"],
            alpha=0.25,
            label="Intervalo",
        )

    ax.axvline(series.index.max(), linestyle="--", linewidth=0.8)
    ax.set_title(title or "Forecast de precios")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio")
    ax.legend()

    return fig


def plot_forecast_plotly(series, forecast_df, title=None, color=None):
    color = color or "#1b4965"

    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        name="Histórico",
        line=dict(color="#888888", width=1.5),
        mode="lines",
    ))

    if "forecast_lower" in forecast_df.columns and "forecast_upper" in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=list(forecast_df["fecha"]) + list(forecast_df["fecha"])[::-1],
            y=list(forecast_df["forecast_upper"]) + list(forecast_df["forecast_lower"])[::-1],
            fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Intervalo 90%",
            showlegend=True,
        ))

    fig.add_trace(go.Scatter(
        x=forecast_df["fecha"],
        y=forecast_df["forecast"],
        name="Forecast",
        line=dict(color=color, width=2.5),
        mode="lines",
    ))

    cutoff = series.index.max()
    fig.add_vline(x=str(cutoff), line_dash="dot", line_color="#aaaaaa", line_width=1.2)

    fig.update_layout(
        title=dict(text=title or "Forecast", font=dict(size=13, color="#333333")),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", zeroline=False),
        legend=dict(orientation="h", y=-0.25, font=dict(size=11)),
        margin=dict(l=40, r=20, t=45, b=60),
        height=300,
    )

    return fig


def business_summary(wide, forecast_df):
    rows = []

    for serie in wide.columns:
        s = wide[serie].dropna()
        fc = forecast_df[forecast_df["serie"] == serie].sort_values("fecha")

        if fc.empty:
            continue

        actual = s.iloc[-1]
        p50_6m = fc["forecast"].iloc[min(25, len(fc) - 1)]
        p50_18m = fc["forecast"].iloc[-1]

        rows.append({
            "serie": serie,
            "actual": round(actual, 2),
            "forecast_6m": round(p50_6m, 2),
            "forecast_18m": round(p50_18m, 2),
            "var_18m_%": round((p50_18m / actual - 1) * 100, 1),
        })

    return rows