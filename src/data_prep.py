from pathlib import Path
import numpy as np
import pandas as pd

PROD_MAP = {
    "Brazil HON — Precio neto USD/kg FCA": "Brazil HON",
    "USA Fillet — Precio neto USD/lb FOB Miami (spot)": "USA Fillet",
}

SERIES_DEF = [
    ("Brazil HON", "10-12 lb"),
    ("Brazil HON", "12-14 lb"),
    ("Brazil HON", "14-16 lb"),
    ("USA Fillet", "2/3"),
    ("USA Fillet", "3/4"),
    ("USA Fillet", "4/5"),
]


def load_raw(path):
    raw = pd.read_csv(path, header=[0, 1, 2])

    lvl0 = pd.Series([c[0] if "Unnamed" not in c[0] else None for c in raw.columns]).ffill()
    lvl1 = pd.Series([c[1] if "Unnamed" not in c[1] else None for c in raw.columns]).ffill()
    lvl2 = [c[2] for c in raw.columns]

    cols = []
    for producto, calibre, col in zip(lvl0, lvl1, lvl2):
        if col in ("Año", "Semana"):
            cols.append(col)
        else:
            cols.append(f"{PROD_MAP.get(producto, producto)}|{calibre}|{col}")

    raw.columns = cols
    raw["Año"] = raw["Año"].astype(int)
    raw["Semana"] = raw["Semana"].astype(int)

    return raw


def iso_week_to_date(year, week):
    return pd.Timestamp.fromisocalendar(int(year), min(int(week), 52), 1)


def build_tidy(path):
    raw = load_raw(path)
    raw["fecha"] = [iso_week_to_date(y, w) for y, w in zip(raw["Año"], raw["Semana"])]

    rec = []

    for cal in ["10-12 lb", "12-14 lb", "14-16 lb"]:
        p1 = pd.to_numeric(raw[f"Brazil HON|{cal}|P Real X"], errors="coerce")
        p2 = pd.to_numeric(raw[f"Brazil HON|{cal}|P Real XII"], errors="coerce")
        precio = pd.concat([p1, p2], axis=1).mean(axis=1)

        for fecha, valor in zip(raw["fecha"], precio):
            rec.append(("Brazil HON", cal, "USD/kg", fecha, valor))

    for cal in ["2/3", "3/4", "4/5"]:
        precio = pd.to_numeric(raw[f"USA Fillet|{cal}|P Real"], errors="coerce")

        for fecha, valor in zip(raw["fecha"], precio):
            rec.append(("USA Fillet", cal, "USD/lb", fecha, valor))

    tidy = pd.DataFrame(
        rec,
        columns=["producto", "calibre", "unidad", "fecha", "precio"]
    )

    return tidy.sort_values(["producto", "calibre", "fecha"]).reset_index(drop=True)


def get_series(tidy, producto, calibre):
    s = (
        tidy[(tidy["producto"] == producto) & (tidy["calibre"] == calibre)]
        .set_index("fecha")["precio"]
        .asfreq("W-MON")
    )

    return s.interpolate("linear", limit=3, limit_area="inside")


def build_wide(tidy):
    wide = pd.DataFrame({
        f"{p}|{c}": get_series(tidy, p, c)
        for p, c in SERIES_DEF
    })

    wide.index.name = "fecha"
    return wide


def prepare_data(path, output_dir="data/processed"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tidy = build_tidy(path)
    wide = build_wide(tidy)

    tidy_clean = (
        wide.reset_index()
        .melt(id_vars="fecha", var_name="serie", value_name="precio")
    )

    tidy_clean[["producto", "calibre"]] = tidy_clean["serie"].str.split("|", expand=True)
    tidy_clean["unidad"] = np.where(tidy_clean["producto"] == "Brazil HON", "USD/kg", "USD/lb")
    tidy_clean = tidy_clean[["fecha", "producto", "calibre", "unidad", "serie", "precio"]]

    tidy_clean.to_parquet(output_dir / "clean_data_long.parquet", index=False)
    tidy_clean.to_csv(output_dir / "clean_data_long.csv", index=False)

    wide.to_parquet(output_dir / "clean_data_wide.parquet")
    wide.to_csv(output_dir / "clean_data_wide.csv")

    return tidy_clean, wide

def append_incremental(base_wide, incremental_wide):
    """
    Agrega nuevas semanas al histórico.
    Si existe una fecha repetida, conserva la última versión.
    """

    updated = pd.concat([base_wide, incremental_wide])

    updated = (
        updated
        .reset_index()
        .drop_duplicates(subset=["fecha"], keep="last")
        .set_index("fecha")
        .sort_index()
    )

    return updated