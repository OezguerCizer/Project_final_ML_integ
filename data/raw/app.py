import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")
st.title("🌍 Wirtschaftliche Verluste & Energieverbrauch in Europa")

# CSV laden
df_type = pd.read_csv("Cleaned_Total_Economic_Losses_Type.csv")
df_year = pd.read_csv("Merged_Energy_Losses.csv")

# Features für die Europakarte (nur 2023-Daten)
map_features = [
    "Geotechnical", "Meteorological", "Hydrological",
    "Climatological (heatwaves)", "Climatological (other)"
]
st.subheader("🗺️ Europakarte für 2023")

map_feature = st.selectbox("Feature auswählen:", map_features)

# Karte (Choropleth) – nur nach Country_ID
if "Country_ID" in df_type.columns and map_feature in df_type.columns:
    fig_map = px.choropleth(
        df_type,
        locations="Country_ID",
        color=map_feature,
        hover_name="Country_name" if "Country_name" in df_type.columns else "Country_ID",
        color_continuous_scale="YlOrRd",
        scope="europe",
        title=f"{map_feature} in Europa (2023)"
    )
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.warning("🌍 Kartendarstellung nicht möglich – überprüfe Country_ID und Feature!")

# 📈 Zeitverlauf
st.subheader("📈 Zeitliche Entwicklung")

country_options = ["Alle"] + sorted(df_year["Country_name"].dropna().unique()) if "Country_name" in df_year.columns else ["Alle"] + sorted(df_year["Country_ID"].dropna().unique())
selected_country = st.selectbox("Land auswählen:", country_options)

all_features = ["Total losses"] + [col for col in df_year.columns if col.startswith(("PEC", "FEC"))]
selected_features = st.multiselect("Features auswählen:", all_features, default=["Total losses"])
chart_type = st.radio("Diagrammtyp:", ["Linie", "Balken"])

df_plot = df_year.copy()
if selected_country != "Alle":
    if "Country_name" in df_plot.columns:
        df_plot = df_plot[df_plot["Country_name"] == selected_country]
    else:
        df_plot = df_plot[df_plot["Country_ID"] == selected_country]

if chart_type == "Linie":
    fig = px.line(df_plot, x="Year", y=selected_features, title="Zeitliche Entwicklung")
else:
    fig = px.bar(df_plot, x="Year", y=selected_features, title="Zeitliche Entwicklung")
st.plotly_chart(fig, use_container_width=True)

# 📊 Vergleich
st.subheader("📊 Ländervergleich")
compare_countries = st.multiselect(
    "Länder zum Vergleich:",
    sorted(df_type["Country_name"].dropna().unique()) if "Country_name" in df_type.columns else sorted(df_type["Country_ID"].dropna().unique())
)
compare_features = st.multiselect("Features zum Vergleich:", map_features, default=["Meteorological"])

if compare_countries and compare_features:
    if "Country_name" in df_type.columns:
        df_compare = df_type[df_type["Country_name"].isin(compare_countries)][["Country_name"] + compare_features]
        st.dataframe(df_compare.set_index("Country_name"))
        csv = df_compare.to_csv(index=False).encode("utf-8")
    else:
        df_compare = df_type[df_type["Country_ID"].isin(compare_countries)][["Country_ID"] + compare_features]
        st.dataframe(df_compare.set_index("Country_ID"))
        csv = df_compare.to_csv(index=False).encode("utf-8")
    st.download_button("📥 CSV exportieren", csv, "vergleich.csv", "text/csv")

