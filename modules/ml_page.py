# modules/ml_page.py — Streamlit-Seite für ML-Features, Training & Forecast
# modules/ml_page.py — ML-UI (autonom, ruft sich selbst auf)
import streamlit as st
import pandas as pd
import plotly.express as px
import ml_core as ml

def render():
    st.header("🔮 ML-Forecast (5 Jahre) – Add-on")

    # Autopilot: Stelle sicher, dass Features/Modelle vorhanden sind
    with st.status("Initialisiere ML-Pipeline …", expanded=False) as status:
        try:
            df_feat, summary = ml.ensure_ready(verbose=False)
            status.update(label="ML bereit ✅", state="complete")
        except Exception as e:
            status.update(label="Fehler bei Initialisierung ❌", state="error")
            st.error(f"Initialisierung fehlgeschlagen: {e}")
            return

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🧱 Features neu erzeugen (NaN → 0)"):
            try:
                df = ml.load_year_df()
                feats = ml.build_features(df)
                st.success(f"Features gespeichert: {ml.FEATURES_CSV} — Zeilen: {len(feats):,}")
            except Exception as e:
                st.error(f"Fehler beim Feature-Build: {e}")

    with c2:
        if st.button("🏋️ Modelle neu trainieren (pro Land)"):
            try:
                df_feat_now = pd.read_csv(ml.FEATURES_CSV)
                res = ml.train_models_per_country(df_feat_now)
                if res.empty:
                    st.warning("Keine Modelle trainiert (zu wenige Jahresdaten pro Land?).")
                else:
                    st.success(f"Modelle gespeichert in {ml.MODELS_DIR}.")
                    st.dataframe(res, width="stretch")
            except Exception as e:
                st.error(f"Training fehlgeschlagen: {e}")

    st.markdown("---")

    # Forecast UI
    try:
        df_feat = pd.read_csv(ml.FEATURES_CSV)
    except Exception as e:
        st.error(f"Features konnten nicht geladen werden: {e}")
        return

    if "Country_ID" not in df_feat.columns or "Year" not in df_feat.columns:
        st.error("In den Features fehlen 'Country_ID' oder 'Year'.")
        return

    countries = sorted(df_feat["Country_ID"].dropna().unique().tolist())
    if not countries:
        st.warning("Keine Länder in den Features gefunden.")
        return

    colA, colB, colC = st.columns([2, 1, 1])
    with colA:
        cid = st.selectbox("Land (Country_ID)", countries)
    with colB:
        scenario = st.selectbox("Szenario", ["Konstant (letzte Werte)", "+2%/Jahr", "-2%/Jahr"])
    with colC:
        horizon = st.slider("Horizont (Jahre)", 1, 10, 5)

    model_pack = ml.load_best_model_for(cid)
    if model_pack is None:
        st.warning("Kein trainiertes Modell gefunden. Bitte Modelle trainieren.")
        return

    try:
        fc = ml.forecast_country(df_feat, cid, model_pack, horizon=horizon, scenario=scenario)
        if fc.empty:
            st.info("Keine Forecast-Daten erzeugt.")
            return

        fig = px.line(fc, x="Year", y="forecast",
                      title=f"Forecast Total losses — {cid} — {scenario}")
        st.plotly_chart(fig, width="stretch")

        st.download_button("📥 Forecast CSV",
                           data=fc.to_csv(index=False).encode("utf-8"),
                           file_name=f"forecast_{cid}.csv",
                           mime="text/csv")
    except Exception as e:
        st.error(f"Forecast fehlgeschlagen: {e}")

# <<< WICHTIG: Beim Laden via exec() direkt rendern
render()
