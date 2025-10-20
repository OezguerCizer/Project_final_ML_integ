import streamlit as st
from pathlib import Path
from modules import ml_page  # <<< NEU: Import der neuen ML-Seite

# Setze Seiteneinstellungen (nur hier erlaubt!)
st.set_page_config(
    page_title="EU-Datenportale",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# --- Sidebar Navigation ---
st.sidebar.title("📊 Standortfaktoren in Europa")
selected_page = st.sidebar.radio("Wähle den gewünschten Themenbereich:", [
    "🌱 Nachhaltigkeit (ESG)",
    "🌍 Emissionen (LCP)",
    "🌪 Wirtschaftliche Verluste",
    "🔮 ML-Forecast"  # <<< NEU: ML-Integration als eigene Seite
])

# --- Seitenstruktur ---
BASE_DIR = Path(__file__).parent
PAGES_DIR = BASE_DIR / "modules"

# --- Logik zum Einbinden der Teil-Dashboards ---
if selected_page == "🌍 Emissionen (LCP)":
    page_path = PAGES_DIR / "LCP_app.py"

elif selected_page == "🌱 Nachhaltigkeit (ESG)":
    page_path = PAGES_DIR / "esg_data_main.py"

elif selected_page == "🌪 Wirtschaftliche Verluste":
    page_path = PAGES_DIR / "app.py"

elif selected_page == "🔮 ML-Forecast":
    page_path = PAGES_DIR / "ml_page.py"

else:
    page_path = None

# --- Ausführung oder Fallback ---
if page_path and page_path.exists():
    code = page_path.read_text(encoding="utf-8")
    exec(code, {})
elif page_path is None and selected_page != "🔮 ML-Forecast":
    st.title("🇪🇺 Europäische Umwelt- & Nachhaltigkeits-Dashboards")
    st.markdown("Erkunde verschiedene Datenportale zu Umwelt, Emissionen, Wirtschaft und Nachhaltigkeit in Europa.")
    st.markdown("---")
