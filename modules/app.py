import os
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

print("Current working directory:", os.getcwd())

st.title("🌍 Wirtschaftliche Verluste & Energieverbrauch in Europa")

DATA_DIRS = [Path("data"), Path(".")]

def _find_file(name: str) -> Path | None:
    for d in DATA_DIRS:
        p = d / name
        if p.exists():
            return p
    return None

@st.cache_data
def load_data():
    p_type = _find_file("Cleaned_Total_Economic_Losses_Type.csv")
    p_year = _find_file("Merged_Energy_Losses.csv")
    if p_type is None or p_year is None:
        raise FileNotFoundError(
            "CSV nicht gefunden. Erwartet:\n"
            "  - data/Cleaned_Total_Economic_Losses_Type.csv\n"
            "  - data/Merged_Energy_Losses.csv\n"
            "oder jeweils im Projektwurzelordner."
        )
    df_type = pd.read_csv(p_type)
    df_year = pd.read_csv(p_year)

    # Spalten trimmen & NaN->0 bei numerischen Werten (wie zuvor gewünscht)
    df_type.columns = df_type.columns.str.strip()
    df_year.columns = df_year.columns.str.strip()
    for df in (df_type, df_year):
        num_cols = df.select_dtypes(include=["number"]).columns
        if len(num_cols):
            df[num_cols] = df[num_cols].fillna(0)
        if "Year" in df.columns:
            df["Year"] = pd.to_numeric(df["Year"], errors="coerce").fillna(0).astype(int)
    return df_type, df_year

df_type, df_year = load_data()

# -----------------------
# 🗺️ Europakarte (2023)
# -----------------------
st.subheader("🗺️ Europakarte für 2023")

map_features = [
    "Geotechnical", "Meteorological", "Hydrological",
    "Climatological (heatwaves)", "Climatological (other)"
]
map_features = [c for c in map_features if c in df_type.columns]
if not map_features:
    st.warning("Keine Typ-Features in den 2023-Daten gefunden.")
else:
    map_feature = st.selectbox("Feature auswählen:", map_features)

    # ISO2 -> ISO3 Mapping
    iso2_to_iso3 = {
        "AT": "AUT", "BE": "BEL", "BG": "BGR", "CY": "CYP", "CZ": "CZE", "DE": "DEU", "DK": "DNK",
        "EE": "EST", "EL": "GRC", "ES": "ESP", "FI": "FIN", "FR": "FRA", "HR": "HRV", "HU": "HUN",
        "IE": "IRL", "IT": "ITA", "LT": "LTU", "LU": "LUX", "LV": "LVA", "MT": "MLT", "NL": "NLD",
        "PL": "POL", "PT": "PRT", "RO": "ROU", "SE": "SWE", "SI": "SVN", "SK": "SVK", "UK": "GBR",
        "NO": "NOR", "IS": "ISL", "LI": "LIE", "CH": "CHE", "TR": "TUR"
    }
    df_type = df_type.copy()
    df_type["Country_ISO3"] = df_type["Country_ID"].map(iso2_to_iso3)

    if "Country_ID" in df_type.columns and map_feature in df_type.columns:
        fig_map = px.choropleth(
            df_type,
            locations="Country_ISO3",
            color=map_feature,
            hover_name="Country_name" if "Country_name" in df_type.columns else "Country_ID",
            color_continuous_scale="YlOrRd",
            scope="europe",
            title=f"{map_feature} in Europa (2023)"
        )
        fig_map.update_layout(margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig_map, width="stretch")
    else:
        st.warning("🌍 Kartendarstellung nicht möglich – überprüfe Country_ID und Feature!")

# -----------------------
# 📈 Zeitliche Entwicklung
# -----------------------
st.subheader("📈 Zeitliche Entwicklung")

name_col = "Country_name" if "Country_name" in df_year.columns else "Country_ID"

country_options = ["Alle"] + sorted(df_year[name_col].dropna().unique())
selected_countries = st.multiselect("Länder auswählen:", country_options, default=["Alle"])

all_features = [col for col in df_year.columns if col.startswith(("PEC", "FEC"))]
if not all_features and "Total losses" in df_year.columns:
    all_features = ["Total losses"]
selected_features = st.multiselect(
    "Features auswählen:",
    all_features,
    default=["FEC total"] if "FEC total" in all_features else all_features[: min(1, len(all_features))]
)

chart_type = st.radio("Diagrammtyp:", ["Linie", "Balken"], horizontal=True)

df_plot = df_year.copy()
# Länderfilter (Mehrfachauswahl)
if "Alle" not in selected_countries:
    df_plot = df_plot[df_plot[name_col].isin(selected_countries)]

# Dynamischer Jahresbereich
if "Year" in df_plot.columns and not df_plot.empty:
    min_year = int(df_plot["Year"].min())
    max_year = int(df_plot["Year"].max())
    if min_year >= max_year:
        year_range = (min_year, max_year)
    else:
        year_range = st.slider("Jahresbereich auswählen:", min_year, max_year, (min_year, max_year))
        df_plot = df_plot[(df_plot["Year"] >= year_range[0]) & (df_plot["Year"] <= year_range[1])]
else:
    st.warning("Kein 'Year' in den Daten gefunden.")
    year_range = None

if selected_features:
    # Wide -> Long für sauberes Plotten mit mehreren Features
    id_vars = ["Year", name_col]
    df_long = df_plot.melt(
        id_vars=id_vars,
        value_vars=selected_features,
        var_name="Feature",
        value_name="Value"
    ).dropna(subset=["Value"])

    if df_long.empty:
        st.info("Keine Daten für die gewählten Filter.")
    else:
        if chart_type == "Linie":
            fig = px.line(
                df_long, x="Year", y="Value",
                color=name_col,
                facet_col="Feature",
                facet_col_wrap=2,
                title="Zeitliche Entwicklung"
            )
        else:
            fig = px.bar(
                df_long, x="Year", y="Value",
                color=name_col,
                facet_col="Feature",
                facet_col_wrap=2,
                title="Zeitliche Entwicklung"
            )
        fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig, width="stretch")
else:
    st.info("Bitte mindestens ein Feature wählen.")

# -----------------------
# 📊 Ländervergleich (2023)
# -----------------------
st.subheader("📊 Ländervergleich (2023)")

compare_pool = df_type["Country_name"].dropna().unique() if "Country_name" in df_type.columns else df_type["Country_ID"].dropna().unique()
compare_countries = st.multiselect(
    "Länder zum Vergleich:",
    sorted(compare_pool)
)

# Für Vergleich nutzen wir die Karten-Features
compare_features = st.multiselect("Features zum Vergleich:", map_features, default=(["Meteorological"] if "Meteorological" in map_features else map_features[:1]))

if compare_countries and compare_features:
    if "Country_name" in df_type.columns:
        df_compare = df_type[df_type["Country_name"].isin(compare_countries)][["Country_name"] + compare_features]
        st.dataframe(df_compare.set_index("Country_name"), width="stretch")
        csv = df_compare.to_csv(index=False).encode("utf-8")
    else:
        df_compare = df_type[df_type["Country_ID"].isin(compare_countries)][["Country_ID"] + compare_features]
        st.dataframe(df_compare.set_index("Country_ID"), width="stretch")
        csv = df_compare.to_csv(index=False).encode("utf-8")

    st.download_button("📥 CSV exportieren", csv, "vergleich.csv", "text/csv")
