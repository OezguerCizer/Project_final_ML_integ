import pandas as pd

# Dateipfade
energy_file = "Energy-consumption_EnEff Proxy 2023_Calculation_RS_V10.xlsx"
type_file = "Cleaned_Total_Economic_Losses_Type.csv"
year_file = "Cleaned_Total_Economic_Losses_Year.csv"
country_file = "Countries.csv"

# Daten laden
df_type = pd.read_csv(type_file)
df_year = pd.read_csv(year_file)
df_countries = pd.read_csv(country_file)
df_energy = pd.read_excel(energy_file, sheet_name="Normalisiert")

# Spaltennamen bereinigen (whitespace entfernen)
for df in [df_type, df_year, df_countries, df_energy]:
    df.columns = df.columns.str.strip()

# Länderinfos auf Country_ID und Country_name beschränken (keine Koordinaten nötig)
if "Country name" in df_countries.columns:
    df_countries.rename(columns={"Country name": "Country_name"}, inplace=True)
df_countries = df_countries[["Country_ID", "Country_name"]]

# Länderinfos zu Economic Loss-Daten hinzufügen
df_type = pd.merge(df_type, df_countries, on="Country_ID", how="left")
df_year = pd.merge(df_year, df_countries, on="Country_ID", how="left")

# PEC & FEC Spalten aus dem Energiedatensatz extrahieren
pec_fec_cols = [col for col in df_energy.columns if col.startswith("PEC") or col.startswith("FEC")]
df_energy_reduced = df_energy[["Country_ID", "Year"] + pec_fec_cols]

# Energiedaten mit Jahresdaten verknüpfen
df_merged = pd.merge(df_year, df_energy_reduced, on=["Country_ID", "Year"], how="left")

# Bereinigte Dateien speichern
df_type.to_csv("Cleaned_Total_Economic_Losses_Type.csv", index=False)
df_merged.to_csv("Merged_Energy_Losses.csv", index=False)

print("✅ Ohne Koordinaten bereinigt und gespeichert!")
