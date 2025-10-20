# Nachhaltigkeit (ESG) – Seite
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# Titel
st.title("Nachhaltigkeit in Europa: Standortfaktoren im Zeit- und Ländervergleich")

# ---------------------------------------
# Daten laden (robust & gecached)
# ---------------------------------------
DATA_DIRS = [Path("data"), Path(".")]

def _find(pathname: str) -> Path | None:
    for d in DATA_DIRS:
        p = d / pathname
        if p.exists():
            return p
    return None

@st.cache_data
def load_data():
    p_water = _find("Water_Exploitation_Index.csv")
    p_eco   = _find("Eco_Innovation_Index.csv")
    p_esg   = _find("ESG_World_Bank_Data_EU-32.csv")
    p_ctry  = _find("Countries.csv")
    if not all([p_water, p_eco, p_esg, p_ctry]):
        raise FileNotFoundError(
            "Erwartete Dateien fehlen. Benötigt:\n"
            "  data/Water_Exploitation_Index.csv\n"
            "  data/Eco_Innovation_Index.csv\n"
            "  data/ESG_World_Bank_Data_EU-32.csv\n"
            "  data/Countries.csv\n"
            "(oder jeweils im Projektwurzelordner)"
        )

    # Semikolon-getrennt
    water = pd.read_csv(p_water, sep=";")
    eco   = pd.read_csv(p_eco,   sep=";")
    esg   = pd.read_csv(p_esg,   sep=";")
    countries = pd.read_csv(p_ctry)

    # Long-Form herstellen
    water["Indicator"] = "Water-Exploitation-Index"
    water = water.rename(columns={"Water-Exploitation-Index": "Value"})
    eco["Indicator"] = "Eco-Innovation-Index"
    eco = eco.rename(columns={"Eco-Innovation-Index": "Value"})
    esg_long = pd.melt(esg, id_vars=["Country_ID", "Year"], var_name="Indicator", value_name="Value")

    # Zusammenführen
    all_data = pd.concat(
        [
            water[["Country_ID", "Year", "Indicator", "Value"]],
            eco[["Country_ID", "Year", "Indicator", "Value"]],
            esg_long
        ],
        ignore_index=True
    )

    # Typen & Säuberung
    all_data["Year"] = pd.to_numeric(all_data["Year"], errors="coerce").fillna(0).astype(int)
    all_data["Value"] = pd.to_numeric(all_data["Value"], errors="coerce")
    countries.columns = countries.columns.str.strip()

    return all_data, countries

df, countries = load_data()
df = df.merge(countries[["Country_ID", "Country name"]], on="Country_ID", how="left")

# Beschreibungen
indikator_beschreibungen = {
    "Adjusted savings: natural resources depletion (% of GNI)": "Natural resource depletion is the sum of net forest depletion, energy depletion, and mineral depletion...",
    "Adjusted savings: net forest depletion (% of GNI)": "Net forest depletion is calculated as the product of unit resource rents and the excess of roundwood harvest over natural growth.",
    "Annual freshwater withdrawals, total (% of internal resources)": "Annual freshwater withdrawals refer to total water withdrawals...",
    "CO2 emissions (metric tons per capita)": "Carbon dioxide emissions are those stemming from the burning of fossil fuels and the manufacture of cement...",
    "Eco-Innovation-Index": "The Eco-Innovation index is calculated based on 12 different indicators...",
    "Economic and Social Rights Performance Score": "Economic and social human rights ensure that all people have access...",
    "Electricity production from coal sources (% of total)": "Sources of electricity refer to the inputs used to generate electricity...",
    "Energy imports, net (% of energy use)": "Net energy imports are estimated as energy use less production...",
    "Fossil fuel energy consumption (% of total)": "Fossil fuel comprises coal, oil, petroleum, and natural gas products.",
    "Heat Index 35": "Total count of days per year where the daily mean Heat Index rose above 35°C.",
    "Heating Degree Days": "A heating degree day (HDD) ist die Anzahl Grad unter 18°C pro Tag...",
    "Land Surface Temperature": "Land Surface Temperature.",
    "Level of water stress: freshwater withdrawal as a proportion of available freshwater resources": "The ratio between total withdrawals and renewable resources...",
    "Methane emissions (metric tons of CO2 equivalent per capita)": "Methane emissions stemming from human activities...",
    "Nitrous oxide emissions (metric tons of CO2 equivalent per capita)": "N2O emissions from agriculture and industry...",
    "PM2.5 air pollution, mean annual exposure (micrograms per cubic meter)": "Population-weighted exposure to ambient PM2.5 pollution...",
    "Renewable energy consumption (% of total final energy consumption)": "Share of renewables in total final energy consumption.",
    "Tree Cover Loss (hectares)": "Year-by-year tree cover loss (not the same as deforestation).",
    "Water-Exploitation-Index": "Vergleicht Wasserentnahme mit erneuerbaren Ressourcen (in %)."
}

st.subheader("🌍 Indikatoren nachhaltiger Entwicklung in Europa")

# ISO2 -> ISO3
iso2_to_iso3 = {
    "AT": "AUT", "BE": "BEL", "BG": "BGR", "CY": "CYP", "CZ": "CZE", "DE": "DEU", "DK": "DNK",
    "EE": "EST", "EL": "GRC", "ES": "ESP", "FI": "FIN", "FR": "FRA", "HR": "HRV", "HU": "HUN",
    "IE": "IRL", "IT": "ITA", "LT": "LTU", "LU": "LUX", "LV": "LVA", "MT": "MLT", "NL": "NLD",
    "PL": "POL", "PT": "PRT", "RO": "ROU", "SE": "SWE", "SI": "SVN", "SK": "SVK", "UK": "GBR",
    "NO": "NOR", "IS": "ISL", "LI": "LIE", "CH": "CHE"
}

# Auswahl für Karte
ind_opts = sorted(df["Indicator"].dropna().unique())
default_ind = "Eco-Innovation-Index" if "Eco-Innovation-Index" in ind_opts else ind_opts[0]
indicator_map = st.selectbox("Indikator auswählen", ind_opts, index=ind_opts.index(default_ind), key="indikator_map")

year_min, year_max = int(df["Year"].min()), int(df["Year"].max())
year_range_map = st.slider("Jahresbereich", min_value=year_min, max_value=year_max, value=(min(2013, year_max), min(2022, year_max)), key="slider_map")

beschreibung = indikator_beschreibungen.get(indicator_map, "No description available.")
st.markdown(f"**Description:** {beschreibung}")

# Karte: Mittelwerte pro Land im Bereich
map_data = (
    df[(df["Indicator"] == indicator_map) & (df["Year"].between(*year_range_map))]
    .groupby("Country_ID", as_index=False)["Value"].mean()
    .rename(columns={"Value": "Mean"})
    .merge(countries, on="Country_ID", how="left")
)
map_data["Country_ISO3"] = map_data["Country_ID"].map(iso2_to_iso3)

fig = px.choropleth(
    map_data,
    locations="Country_ISO3",
    locationmode="ISO-3",
    color="Mean",
    hover_name="Country name",
    color_continuous_scale="YlOrRd",
    scope="europe",          # falls du Weltansicht willst: "world"
    title=f"{indicator_map} – Mittelwert {year_range_map[0]}–{year_range_map[1]}"
)
st.plotly_chart(fig, width="stretch")

st.write(map_data[["Country_ID", "Mean"]].sort_values(by="Mean", ascending=False))

st.divider()

# ------------------------------
# 📈 Zeitverlauf
# ------------------------------
st.subheader("📈 Umweltindikatoren im Zeitverlauf")

indicator_zeit = st.selectbox("Indikator auswählen", ind_opts, index=ind_opts.index(default_ind), key="indikator_zeit")
countries_pool = sorted(df["Country name"].dropna().unique())
countries_zeit = st.multiselect("Länder auswählen", countries_pool, default=[c for c in ["Germany", "Denmark"] if c in countries_pool], key="länder_zeit")
year_range_zeit = st.slider("Jahresbereich", min_value=year_min, max_value=year_max, value=(min(2013, year_max), min(2022, year_max)), key="slider_zeit")

beschreibung = indikator_beschreibungen.get(indicator_zeit, "No description available.")
st.markdown(f"**Description:** {beschreibung}")

df_zeitverlauf = df[
    (df["Indicator"] == indicator_zeit) &
    (df["Country name"].isin(countries_zeit)) &
    (df["Year"].between(*year_range_zeit))
].copy()

chart_type = st.radio("Diagrammtyp auswählen", ["Linie", "Balken"], key="chart_type_zeit", horizontal=True)

if not df_zeitverlauf.empty:
    if chart_type == "Linie":
        fig = px.line(
            df_zeitverlauf, x="Year", y="Value", color="Country name",
            markers=True,
            title=f"{indicator_zeit} ({year_range_zeit[0]}–{year_range_zeit[1]})",
            labels={"Year": "Year", "Value": "Value", "Country name": "Country"}
        )
    else:
        fig = px.bar(
            df_zeitverlauf, x="Year", y="Value", color="Country name",
            barmode="group",
            title=f"{indicator_zeit} ({year_range_zeit[0]}–{year_range_zeit[1]})",
            labels={"Year": "Year", "Value": "Value", "Country name": "Country"}
        )
    st.plotly_chart(fig, width="stretch")
else:
    st.info("Keine Daten für die aktuelle Auswahl.")

st.divider()

# ------------------------------
# 🏆 Länderranking
# ------------------------------
st.subheader("🏆 Länderranking")

year_options = ["Mean"] + sorted(df["Year"].unique().tolist())
year_sel = st.selectbox("Jahr auswählen", year_options)
indicator_sel = st.selectbox("Indikator auswählen", ind_opts, index=ind_opts.index(default_ind))
amount_countries = st.slider("Anzahl der Länder festlegen", min_value=3, max_value=10, value=5)

if year_sel == "Mean":
    grouped = df[df["Indicator"] == indicator_sel].groupby("Country name", as_index=False)["Value"].mean()
    top_countries = grouped.sort_values(by="Value", ascending=False).head(amount_countries)
else:
    filtered = df[(df["Year"] == year_sel) & (df["Indicator"] == indicator_sel)]
    top_countries = filtered.sort_values(by="Value", ascending=False).head(amount_countries)

if not top_countries.empty:
    fig = px.bar(
        top_countries,
        x="Value", y="Country name",
        orientation="h",
        title=f"Top {amount_countries} Länder ({indicator_sel}) im Jahr {year_sel}",
        labels={"Value": "Value", "Country name": "Country"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, width="stretch")
else:
    st.write("No data available.")

st.divider()

# ------------------------------
# 📊 Ländervergleich (Pivot)
# ------------------------------
st.subheader("📊 Ländervergleich nach Nachhaltigkeitskriterien")

indicator_vergleich = st.selectbox("Indikator auswählen", ind_opts, index=ind_opts.index(default_ind), key="indikator_vergleich")
countries_vergleich = st.multiselect("Länder auswählen", countries_pool, default=[c for c in ["Germany", "Denmark"] if c in countries_pool], key="länder_vergleich")

df_vergleich = df[(df["Indicator"] == indicator_vergleich) & (df["Country name"].isin(countries_vergleich))].copy()
df_pivot = df_vergleich.pivot_table(index="Year", columns="Country name", values="Value")

st.dataframe(df_pivot, width="stretch")

st.divider()

st.write("Fehlende Werte werden als NaN angezeigt (keine Mittelwert-Imputation).")

st.download_button(
    label="Daten des Ländervergleichs als CSV herunterladen",
    data=df_pivot.to_csv().encode("utf-8"),
    file_name="sustainability_data.csv",
    mime="text/csv"
)
