# modules/LCP_app.py  —  Luft- & Wasser-Emissionen (robust & modernisiert)
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Emissionen in Europa: Luft- und Wasserschadstoffe")

# -----------------------------
# Dateien robust laden (cached)
# -----------------------------
DATA_DIRS = [Path("data"), Path(".")]

def _find(pathname: str) -> Path | None:
    for d in DATA_DIRS:
        p = d / pathname
        if p.exists():
            return p
    return None

@st.cache_data
def load_data():
    p_air   = _find("F1_1_Air_Releases_National.csv")
    p_water = _find("F2_1_Water_Releases_National.csv")
    p_ctry  = _find("Countries.csv")
    if not all([p_air, p_water, p_ctry]):
        raise FileNotFoundError(
            "Erwartete Dateien fehlen. Benötigt:\n"
            "  data/F1_1_Air_Releases_National.csv\n"
            "  data/F2_1_Water_Releases_National.csv\n"
            "  data/Countries.csv\n"
            "(oder jeweils im Projektwurzelordner)"
        )

    air = pd.read_csv(p_air)
    water = pd.read_csv(p_water)
    countries = pd.read_csv(p_ctry)

    # Spalten vereinheitlichen
    for df in (air, water):
        df.rename(columns={"reportingYear": "Year", "countryName": "CountryName"}, inplace=True)
        df.columns = df.columns.str.strip()
        # Numerik erzwingen
        if "Year" in df.columns:
            df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        if "Releases" in df.columns:
            df["Releases"] = pd.to_numeric(df["Releases"], errors="coerce")

        # Pollutant als string
        if "Pollutant" in df.columns:
            df["Pollutant"] = df["Pollutant"].astype(str)

    # Countries sauber
    countries.columns = countries.columns.str.strip()

    return air, water, countries

air, water, countries = load_data()

# ISO2 → ISO3 Mapping für Choropleth
iso2_to_iso3 = {
    "AT": "AUT", "BE": "BEL", "BG": "BGR", "CY": "CYP", "CZ": "CZE", "DE": "DEU", "DK": "DNK",
    "EE": "EST", "EL": "GRC", "ES": "ESP", "FI": "FIN", "FR": "FRA", "HR": "HRV", "HU": "HUN",
    "IE": "IRL", "IT": "ITA", "LT": "LTU", "LU": "LUX", "LV": "LVA", "MT": "MLT", "NL": "NLD",
    "PL": "POL", "PT": "PRT", "RO": "ROU", "SE": "SWE", "SI": "SVN", "SK": "SVK", "UK": "GBR",
    "NO": "NOR", "IS": "ISL", "LI": "LIE", "CH": "CHE"
}
countries["Country_ISO3"] = countries["Country_ID"].map(iso2_to_iso3)

# Country_ID in Air/Water mappen (über Countries.csv)
if "Country name" in countries.columns:
    id_map = countries.set_index("Country name")["Country_ID"].to_dict()
else:
    # Fallback: versuche über CountryName Spalte in den Emissionsdatensätzen
    id_map = {}

for df in (air, water):
    if "CountryName" in df.columns:
        df["Country_ID"] = df["CountryName"].map(id_map)
    # Lücken bei Year/Releases beheben
    if "Year" in df.columns:
        df["Year"] = df["Year"].fillna(method="ffill").fillna(method="bfill")
        df["Year"] = df["Year"].astype(int)
    if "Releases" in df.columns:
        df["Releases"] = df["Releases"].fillna(0)

# -----------------------------
# UI: Emissionstyp & Schadstoff
# -----------------------------
emission_type = st.radio("Emissionstyp auswählen", ["Air", "Water"], horizontal=True)
df_base = air if emission_type == "Air" else water

if df_base.empty:
    st.warning("Keine Daten im ausgewählten Datensatz.")
    st.stop()

pollutants = sorted(df_base["Pollutant"].dropna().unique().tolist())
if not pollutants:
    st.warning("Keine 'Pollutant'-Werte vorhanden.")
    st.stop()

pollutant = st.selectbox("Schadstoff auswählen", pollutants)

# Gefilterte Basis
df = df_base[df_base["Pollutant"] == pollutant].copy()
if df.empty:
    st.info("Keine Zeilen für den gewählten Schadstoff.")
    st.stop()

# -----------------------------
# 🌍 Karte: Durchschnitt pro Land
# -----------------------------
st.subheader("🌍 Durchschnittliche Emissionen nach Land")

year_min = int(df["Year"].min())
year_max = int(df["Year"].max())
jahrbereich_map = st.slider("Jahresbereich", min_value=year_min, max_value=year_max,
                            value=(max(year_min, year_max-5), year_max), key="jahr_slider_map")

map_data = (
    df[df["Year"].between(*jahrbereich_map)]
    .groupby("Country_ID", as_index=False)["Releases"].mean()
    .rename(columns={"Releases": "Mean"})
    .merge(countries, on="Country_ID", how="left")
)

if map_data.empty:
    st.info("Keine Daten im gewählten Zeitraum.")
else:
    fig = px.choropleth(
        map_data,
        locations="Country_ISO3",
        locationmode="ISO-3",
        color="Mean",
        hover_name="Country name",
        color_continuous_scale="YlOrRd",
        scope="europe",
        title=f"{pollutant} – Mittelwert {jahrbereich_map[0]}–{jahrbereich_map[1]}",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig, width="stretch")

    st.dataframe(map_data[["Country name", "Mean"]].sort_values(by="Mean", ascending=False), width="stretch")

st.divider()

# -----------------------------
# 📈 Zeitverlauf
# -----------------------------
st.subheader("📈 Emissionen im Zeitverlauf")

available_countries = sorted(countries["Country name"].dropna().unique().tolist())
default_countries = [c for c in ["Germany", "France"] if c in available_countries]
länder = st.multiselect("Länder auswählen", available_countries, default=default_countries)

jahrbereich_ts = st.slider("Jahresbereich", min_value=year_min, max_value=year_max,
                           value=(max(year_min, year_max-5), year_max))
chart_type = st.radio("Diagrammtyp auswählen", ["Linie", "Balken"], horizontal=True)

df_line = df[df["CountryName"].isin(länder) & df["Year"].between(*jahrbereich_ts)].copy()
df_line = df_line.sort_values(by=["CountryName", "Year"])

if df_line.empty:
    st.info("Keine Daten für die gewählte Auswahl.")
else:
    if chart_type == "Linie":
        fig_line = px.line(
            df_line, x="Year", y="Releases", color="CountryName",
            markers=True,
            title=f"{pollutant} ({jahrbereich_ts[0]}–{jahrbereich_ts[1]})"
        )
        fig_line.update_traces(connectgaps=True)
    else:
        fig_line = px.bar(
            df_line, x="Year", y="Releases", color="CountryName",
            title=f"{pollutant} ({jahrbereich_ts[0]}–{jahrbereich_ts[1]})"
        )
    fig_line.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig_line, width="stretch")

st.divider()

# -----------------------------
# 🏆 Länderranking
# -----------------------------
st.subheader("🏆 Länderranking")

year_options = ["Mean"] + sorted(df["Year"].unique().tolist())
year_sel = st.selectbox("Jahr auswählen", year_options)
top_n = st.slider("Anzahl Top-Länder", 3, 10, 5)

if year_sel == "Mean":
    ranking = df.groupby("CountryName", as_index=False)["Releases"].mean()
else:
    ranking = df[df["Year"] == year_sel].groupby("CountryName", as_index=False)["Releases"].sum()

top_countries = ranking.sort_values(by="Releases", ascending=False).head(top_n)

if top_countries.empty:
    st.info("Keine Daten für das Ranking.")
else:
    fig_bar = px.bar(top_countries, x="Releases", y="CountryName", orientation="h",
                     title=f"Top {top_n} – {pollutant} ({'Mittelwert' if year_sel=='Mean' else year_sel})")
    fig_bar.update_layout(yaxis=dict(categoryorder="total ascending"), margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig_bar, width="stretch")

st.divider()

# -----------------------------
# 📊 Vergleich ausgewählter Länder
# -----------------------------
st.subheader("📊 Vergleich ausgewählter Länder")

länder_vergleich = st.multiselect("Länder für Vergleich", available_countries, default=default_countries)
df_vergleich = df[df["CountryName"].isin(länder_vergleich)].copy()
df_pivot = pd.pivot_table(df_vergleich, index="Year", columns="CountryName", values="Releases", aggfunc="sum")

st.dataframe(df_pivot, width="stretch")
st.download_button(
    "CSV-Export",
    data=df_pivot.to_csv().encode("utf-8"),
    file_name="emissions_vergleich.csv",
    mime="text/csv"
)
