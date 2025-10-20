# data_prep.py (gefixt)
import pandas as pd
from pathlib import Path
import pandas.api.types as ptypes

# Wir suchen automatisch in '.', './data', './datasets'
SEARCH_DIRS = [Path("."), Path("data"), Path("datasets")]

def find_file(candidates: list[str]) -> Path | None:
    for d in SEARCH_DIRS:
        for name in candidates:
            p = d / name
            if p.exists():
                return p.resolve()
    return None

# Mögliche Dateinamen (falls du mal andere Namen probierst)
YEAR_CANDIDATES = [
    "Merged_Energy_Losses.csv",
    "merged_energy_losses.csv",
    "Merged_Energy_Losses_clean.csv",
]
TYPE_CANDIDATES = [
    "Cleaned_Total_Economic_Losses_Type.csv",
    "cleaned_total_economic_losses_type.csv",
]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ------------------------
    # 1) Spalten-Whitespaces entfernen (nur Textspalten!)
    # ------------------------
    for col in df.columns:
        if ptypes.is_string_dtype(df[col]):
            df[col] = df[col].astype("string").str.strip()

    # ------------------------
    # 2) Year als int, NaN->0
    # ------------------------
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").fillna(0).astype(int)

    # ------------------------
    # 3) Ziel prüfen
    # ------------------------
    if "Total losses" not in df.columns:
        raise ValueError("Spalte 'Total losses' fehlt im Jahresdatensatz (Merged_Energy_Losses.csv).")

    # ------------------------
    # 4) Numerische NaN -> 0 (KEINE Mittelwert-Imputation)
    # ------------------------
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    # ------------------------
    # 5) Sortieren & Feature-Engineering pro Land
    # ------------------------
    if "Country_ID" not in df.columns:
        raise ValueError("Spalte 'Country_ID' fehlt im Jahresdatensatz (wird für Gruppenbildung benötigt).")

    df = df.sort_values(["Country_ID", "Year"])

    def add_feats(gr: pd.DataFrame) -> pd.DataFrame:
        g = gr.copy()
        # Lags
        g["loss_lag1"] = g["Total losses"].shift(1).fillna(0)
        g["loss_lag2"] = g["Total losses"].shift(2).fillna(0)
        g["loss_lag3"] = g["Total losses"].shift(3).fillna(0)
        # Rolling
        g["loss_roll3_mean"] = g["Total losses"].rolling(3, min_periods=1).mean()
        g["loss_roll5_mean"] = g["Total losses"].rolling(5, min_periods=1).mean()
        # Trends
        g["year_centered"] = g["Year"] - g["Year"].min()
        g["year_sq"] = g["year_centered"] ** 2
        return g

    df = df.groupby("Country_ID", group_keys=False).apply(add_feats)

    # ------------------------
    # 6) Spaltenauswahl
    # ------------------------
    base_feats = [
        "Year", "year_centered", "year_sq",
        "loss_lag1", "loss_lag2", "loss_lag3",
        "loss_roll3_mean", "loss_roll5_mean"
    ]
    exog = [c for c in df.columns if c.startswith(("PEC", "FEC"))]

    keep = (["Country_ID"] + (["Country_name"] if "Country_name" in df.columns else [])
            + base_feats + exog + ["Total losses"])
    return df[keep].copy()

def main():
    # Dateien finden
    def _print_candidates(label, cands):
        return f"{label}: " + ", ".join(cands)

    year_path = find_file(YEAR_CANDIDATES)
    type_path = find_file(TYPE_CANDIDATES)

    if year_path is None:
        raise AssertionError(
            "Merged_Energy_Losses.csv nicht gefunden.\n"
            "Lege die Datei ins Projektverzeichnis ODER nach ./data.\n"
            + _print_candidates("Gesuchte Namen", YEAR_CANDIDATES)
        )
    print(f"✅ Jahresdaten gefunden: {year_path}")

    if type_path is None:
        print("ℹ️ Typdaten (2023) nicht gefunden – ist ok für ML-Features, wird übersprungen.")
    else:
        print(f"ℹ️ Typdaten gefunden: {type_path}")

    # Laden + Features bauen
    df_year = pd.read_csv(year_path)
    feats = build_features(df_year)
    feats.to_csv("features_for_ml.csv", index=False)
    print(f"✅ Features gespeichert: features_for_ml.csv  | Zeilen: {len(feats)} | Spalten: {len(feats.columns)}")

if __name__ == "__main__":
    main()
