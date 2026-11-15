"""
FireTrackers — MODIS + VIIRS | Global Fire Activity Explorer
NASA Space Apps Challenge 2026 — Harmonization of MODIS and VIIRS Hot Spots

Geovisor Map-First:
1. Vista inicial: El mapa ocupa la pantalla principal. No hay estadísticas previas.
2. Selección de área: El usuario navega libremente (paneo/zoom sin recálculos).
   El análisis se dispara únicamente al completar la delimitación de una zona.
3. Panel contextual: Al seleccionar una zona se despliega un panel lateral con
   KPIs, calendario histórico (heatmap), anomalías, tendencias y descarga CSV,
   manteniendo el mapa visible. Puede cerrarse con una ✕ para volver al mapa completo.
4. Información interna oculta: Sin conceptos de desarrollo en la interfaz pública.
"""

from pathlib import Path
import sys
import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import folium
from folium.plugins import Draw, FastMarkerCluster
from streamlit_folium import st_folium
from shapely.geometry import shape

# ── Importación de módulos del proyecto ───────────────────────────────────────
SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analytics_core import (
    validate_coordinates,
    validate_dates,
    filter_bbox,
    filter_dates,
    monthly_counts,
    calendar_matrix,
    historical_median,
    compute_anomalies,
    sanitize_for_csv,
    MESES_ES,
    NOMBRES_MESES,
    ANOMALY_THRESHOLD_HIGH,
    ANOMALY_THRESHOLD_LOW,
)

from temporal_play import render_player
from activity_zones import render_activity_zones

from pipeline_adapter import (
    request_pipeline_execution, request_ingest_preview, request_full_pipeline,
    is_bbox_covered_by_colombia, FECHA_MIN_COLOMBIA, FECHA_MAX_COLOMBIA,
)

# ── Rutas y Constantes ────────────────────────────────────────────────────────
DATA_DIR = BASE_DIR / "data"
HARMONIZED_DIR = DATA_DIR / "harmonized"

HEATMAP_COLORSCALE = "YlOrRd"
ANOMALY_COLOR_MAP = {
    "alta": "#D32F2F",
    "alta (mediana 0)": "#E65100",
    "normal": "#2E7D32",
    "baja": "#1976D2",
    "sin_referencia": "#9E9E9E",
}
COLOR_VIIRS = "#D32F2F"
COLOR_MODIS = "#F57C00"

MAX_MAP_DETECTIONS = 3_000
CLICK_SEARCH_RADIUS_DEG = 0.05

REGIONES_PRESET = {
    "Zona Norte": (-77.50, 7.80, -71.00, 12.56),
    "Zona Centro": (-77.20, 1.50, -72.50, 8.50),
    "Zona Sur": (-75.50, -4.23, -69.50, 1.50),
    "Zona Oriente": (-73.50, 1.50, -67.00, 6.50),
    "Zona Occidente": (-79.20, 1.00, -76.20, 7.50),
    "Cobertura Base Completa": (-81.71, -4.23, -66.90, 12.56),
}

# ── Configuración de Página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="FireTrackers — Global Fire Activity Explorer",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Estilos CSS profesionales: Minimalistas, sobrios y Map-First
st.markdown("""
<style>
    /* Ajuste de contenedor principal para máxima visibilidad */
    .block-container {
        padding-top: 0.5rem;
        padding-bottom: 0.8rem;
        padding-left: 1.2rem;
        padding-right: 1.2rem;
        max-width: 100%;
    }

    /* Barra superior profesional */
    .top-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 4px 0 10px 0;
        margin-bottom: 8px;
        border-bottom: 1px solid #e2e8f0;
    }
    .top-brand {
        display: flex;
        align-items: baseline;
        gap: 10px;
    }
    .top-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.03em;
        margin: 0;
    }
    .top-subtitle {
        font-size: 0.82rem;
        color: #64748b;
        font-weight: 500;
        margin: 0;
    }
    .top-badge {
        font-size: 0.74rem;
        background: #f1f5f9;
        color: #475569;
        padding: 3px 10px;
        border-radius: 9999px;
        border: 1px solid #cbd5e1;
        font-weight: 500;
    }

    /* Barra de herramientas compacta */
    .toolbar-container {
        display: flex;
        align-items: center;
        gap: 12px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 6px 14px;
        margin-bottom: 10px;
        font-size: 0.84rem;
    }

    /* Banner informativo inicial cuando no hay selección */
    .prompt-banner {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #f8fafc;
        border: 1px dashed #cbd5e1;
        border-radius: 6px;
        padding: 8px 14px;
        margin-bottom: 10px;
        font-size: 0.84rem;
        color: #334155;
    }

    /* Panel contextual lateral derecho */
    .panel-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 8px;
        margin-bottom: 10px;
        border-bottom: 1px solid #e2e8f0;
    }
    .panel-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
    }
    .panel-subtitle {
        font-size: 0.78rem;
        color: #64748b;
        margin: 2px 0 0 0;
    }

    /* Tarjetas métricas compactas dentro del panel */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 8px 10px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.74rem !important;
        color: #64748b !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.18rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
    }

    /* Contenedor Folium */
    .leaflet-container {
        border-radius: 8px;
        border: 1px solid #cbd5e1;
        box-shadow: 0 1px 6px rgba(0,0,0,0.06);
    }

    /* Pestañas compactas */
    [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid #e2e8f0;
    }
    [data-baseweb="tab"] {
        font-size: 0.80rem;
        padding: 5px 10px;
    }
</style>
""", unsafe_allow_html=True)



# Estilo FireTrackers · capa visual. No altera pipelines ni estadisticas.
st.markdown("""
<style>
:root { --ft-ink:#18263b; --ft-orange:#ff7b2b; --ft-border:rgba(255,255,255,.68); }
.stApp {background:linear-gradient(140deg,#d5ecf8 0%,#f1f6fc 52%,#eaf5f3 100%);color:var(--ft-ink);}
[data-testid="stHeader"] {background:transparent !important;height:1.4rem;}
.block-container {padding-top:.30rem !important;padding-left:.9rem !important;padding-right:.9rem !important;max-width:100% !important;}
.top-navbar {background:rgba(255,255,255,.76);backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);border:1px solid var(--ft-border);border-radius:22px;box-shadow:0 10px 30px rgba(32,69,100,.12);padding:14px 22px;margin-bottom:14px;}
.top-title {font-size:1.53rem;color:#10233c;letter-spacing:-.035em;}
.top-subtitle {font-size:.79rem;color:#60728b;}
.top-badge {background:rgba(226,237,247,.64);border:1px solid #cbdceb;color:#425c75;font-size:.70rem;padding:6px 12px;}
[data-testid="stHorizontalBlock"] {align-items:stretch;}
[data-testid="stVerticalBlockBorderWrapper"] > div {border-radius:18px;}
.stButton > button, .stDownloadButton > button {border-radius:13px !important;border:1px solid rgba(190,205,220,.85) !important;background:rgba(255,255,255,.83) !important;box-shadow:0 3px 12px rgba(48,70,96,.07) !important;color:#25334b !important;min-height:2.6rem;}
.stButton > button:hover,.stDownloadButton > button:hover {border-color:#ffad7a !important;background:#fff6f0 !important;color:#d75116 !important;}
.stButton > button[kind="primary"],.stDownloadButton > button[kind="primary"]{background:linear-gradient(100deg,#ff8c39,#fb6524) !important;color:white !important;border:0 !important;}
[data-testid="stMetric"] {border-radius:18px !important;background:linear-gradient(125deg,rgba(255,255,255,.94),rgba(255,255,255,.70)) !important;border:1px solid rgba(255,255,255,.9) !important;box-shadow:0 5px 22px rgba(33,64,96,.10) !important;padding:15px 15px !important;}
[data-testid="stMetricValue"] {font-size:1.52rem !important;color:#172842 !important;}
[data-testid="stMetricLabel"] {color:#61728a !important;}
[data-baseweb="tab-list"] {background:rgba(255,255,255,.73);border:1px solid rgba(205,220,231,.8);border-radius:14px;display:flex;gap:2px;padding:5px;margin:12px 0 10px;}
[data-baseweb="tab"] {border-radius:10px !important;padding:10px 11px !important;color:#3d5067;}
[data-baseweb="tab"][aria-selected="true"] {background:linear-gradient(100deg,#fff0e5,#ffe2d2) !important;color:#c3480b !important;border-bottom:0 !important;font-weight:700;}
[data-testid="stDataFrame"] {border:1px solid #dce7f0;border-radius:13px;overflow:hidden;}
[data-testid="stAlert"] {border-radius:14px !important;}
[data-testid="stDateInput"] > div, [data-testid="stSelectbox"] > div {border-radius:12px;}
.leaflet-container {border-radius:19px !important;border:1px solid rgba(255,255,255,.8) !important;box-shadow:0 8px 30px rgba(24,58,88,.13) !important;}
.panel-header{background:rgba(255,255,255,.63);border-radius:15px;border:1px solid rgba(255,255,255,.8);padding:12px 14px;margin-bottom:8px;}
.panel-title{font-size:1.34rem !important;color:#12253f !important;}
.panel-subtitle{font-size:.81rem;color:#62768d;}
.prompt-banner {border:1px solid #c4dfef;background:rgba(255,255,255,.65);border-radius:13px;}
@media(max-width:1050px){.top-navbar{padding:10px 12px;border-radius:17px}.top-subtitle{display:none}.top-badge{display:none}[data-baseweb="tab"]{padding:8px 6px !important;font-size:.7rem !important}}
</style>
""", unsafe_allow_html=True)


# FireTrackers V2: mapa de fondo y panel geoespacial flotante.
# La modificación es exclusivamente de presentación.
st.markdown("""
<style>
.stApp { background:#d4e8f5 !important; }
[data-testid="stHeader"]{background:transparent !important;height:1.4rem !important}
.block-container{padding-top:0.15rem !important;padding-left:.65rem !important;padding-right:.65rem !important;padding-bottom:.65rem !important}
.top-navbar{position:relative;z-index:20;margin-bottom:8px !important;padding:10px 18px !important;border-radius:23px !important;background:rgba(248,252,255,.79) !important;border:1px solid rgba(255,255,255,.93) !important;box-shadow:0 10px 26px rgba(37,78,116,.12) !important;min-height:66px}
.top-title{font-size:1.65rem !important}.top-subtitle{font-size:.76rem !important}
/* Barra de controles: una superficie de vidrio, no cinco tarjetas aisladas. */
[data-testid="stHorizontalBlock"]:has([data-testid="stDateInput"]){background:rgba(249,252,255,.69);backdrop-filter:blur(16px);border:1px solid rgba(255,255,255,.9);border-radius:18px;padding:7px 10px;margin-bottom:8px;box-shadow:0 8px 22px rgba(33,68,99,.09);align-items:center}
[data-testid="stDateInput"] [data-baseweb="input"], [data-testid="stSelectbox"] [data-baseweb="select"]{background:rgba(255,255,255,.72) !important;border-radius:14px !important}
/* Elementos de Folium: la propia superficie geográfica ocupa el ancho libre. */
[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .ft-overlay-anchor){display:block !important;position:relative !important;min-height:720px;margin-top:5px;isolation:isolate}
[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .ft-overlay-anchor) > [data-testid="stColumn"]:first-child{width:100% !important;max-width:100% !important;min-width:100% !important;flex:1 1 100% !important;position:relative;z-index:1}
[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .ft-overlay-anchor) > [data-testid="stColumn"]:nth-child(2){position:absolute !important;z-index:10;right:12px;top:12px;bottom:12px;width:min(510px,42vw) !important;max-width:min(510px,42vw) !important;min-width:350px !important;flex:none !important;border:1px solid rgba(255,255,255,.89);border-radius:27px;padding:14px 14px 16px;background:linear-gradient(135deg,rgba(250,253,255,.93),rgba(226,240,250,.91));backdrop-filter:blur(22px);-webkit-backdrop-filter:blur(22px);box-shadow:0 15px 55px rgba(15,54,91,.22);overflow-y:auto;max-height:700px;scrollbar-width:thin}
.ft-overlay-anchor{display:none}
/* Al cargar sin panel, el mapa sigue siendo protagonista. */
[data-testid="stFolium"] iframe{border-radius:22px !important}
iframe[title*="streamlit_folium"], iframe[title*="folium"]{border-radius:22px !important}
.panel-header{background:transparent !important;border:0 !important;border-bottom:1px solid rgba(148,170,191,.30) !important;border-radius:0 !important;box-shadow:none !important;padding:8px 2px 12px !important}
.panel-title{font-size:1.48rem !important;font-weight:800 !important}
.panel-subtitle{font-size:.75rem !important}
[data-testid="stMetric"]{background:rgba(255,255,255,.82) !important;border-radius:17px !important;padding:13px 10px !important;box-shadow:0 4px 16px rgba(65,100,132,.08) !important}
[data-testid="stMetricValue"]{font-size:1.65rem !important}
[data-baseweb="tab-list"]{background:rgba(255,255,255,.85) !important;border-radius:16px !important;gap:0 !important;overflow-x:auto}
[data-baseweb="tab"]{font-size:.70rem !important;padding:8px 6px !important;white-space:nowrap !important}
[data-baseweb="tab"][aria-selected="true"]{background:#ff7e35 !important;color:white !important;border-radius:12px !important}
[data-testid="stPlotlyChart"]{background:rgba(255,255,255,.8);border-radius:15px;padding:5px}
[data-testid="stDataFrame"]{background:rgba(255,255,255,.8);border-radius:15px !important}
[data-testid="stDownloadButton"] button{background:#1b2e48 !important;color:white !important;border:0 !important}
/* Evita superposiciones inservibles en portátil y móvil. */
@media(max-width:1050px){
 [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .ft-overlay-anchor) > [data-testid="stColumn"]:nth-child(2){width:47vw !important;max-width:47vw !important;min-width:320px !important;}
}
@media(max-width:760px){
 [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .ft-overlay-anchor){display:flex !important;flex-direction:column !important;gap:12px;min-height:0}
 [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .ft-overlay-anchor) > [data-testid="stColumn"]:nth-child(2){position:relative !important;top:auto !important;right:auto !important;bottom:auto !important;width:100% !important;max-width:100% !important;min-width:0 !important;max-height:unset !important}
 .top-subtitle,.top-badge{display:none !important}
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS HISTÓRICOS (Base de Referencia)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_base_dataset():
    csv_path = HARMONIZED_DIR / "incendios_armonizados.csv"
    if not csv_path.exists():
        return None, "Dataset no encontrado."

    dtypes = {
        "latitude": "float32",
        "longitude": "float32",
        "acq_time": "int16",
        "satellite": "category",
        "instrument": "category",
        "confidence": "str",
        "type": "int8",
        "frp": "float32",
        "daynight": "category",
        "source": "category",
        "year": "int16",
        "confidence_level": "category",
        "confidence_num": "int8",
        "priority": "int8",
    }
    try:
        df = pd.read_csv(csv_path, dtype=dtypes, parse_dates=["timestamp"])
        df["month"] = df["timestamp"].dt.month.astype("int8")
        df["month_name"] = df["month"].map(MESES_ES).astype("category")
        df["year_month"] = df["timestamp"].dt.to_period("M").astype(str)
        return df, ""
    except Exception as err:
        return None, f"Error de lectura: {err}"


df_raw, load_err = load_base_dataset()
if df_raw is None or df_raw.empty:
    st.error(f"Error cargando datos: {load_err}")
    st.stop()

DS_MIN_DATE = datetime.date(2000, 11, 1)
DS_MAX_DATE = datetime.date.today()


# ══════════════════════════════════════════════════════════════════════════════
# GESTIÓN DEL ESTADO DE SESIÓN (MAP-FIRST)
# ══════════════════════════════════════════════════════════════════════════════
# En el inicio, active_selection es None -> Muestra SOLO el mapa.
if "active_selection" not in st.session_state:
    st.session_state["active_selection"] = None  # None = Modo inicial limpio
if "show_panel" not in st.session_state:
    st.session_state["show_panel"] = False
if "map_center" not in st.session_state:
    st.session_state["map_center"] = [4.5, -73.5]
if "map_zoom" not in st.session_state:
    st.session_state["map_zoom"] = 6
if "date_range" not in st.session_state:
    st.session_state["date_range"] = (datetime.date.today() - datetime.timedelta(days=2), datetime.date.today())
if "sel_sources" not in st.session_state:
    st.session_state["sel_sources"] = sorted(df_raw["source"].cat.categories.tolist())
if "sel_conf" not in st.session_state:
    st.session_state["sel_conf"] = sorted(df_raw["confidence_level"].cat.categories.tolist())
if "selected_point" not in st.session_state:
    st.session_state["selected_point"] = None
if "_last_drawing_id" not in st.session_state:
    st.session_state["_last_drawing_id"] = None
if "show_reference_layer" not in st.session_state:
    st.session_state["show_reference_layer"] = True


# ══════════════════════════════════════════════════════════════════════════════
# ENCABEZADO SUPERIOR COMPACTO (FireTrackers)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="top-navbar">
    <div class="top-brand">
        <h1 class="top-title">🔥 FireTrackers</h1>
        <p class="top-subtitle">MODIS + VIIRS | Global Fire Activity Explorer</p>
    </div>
    <div>
        <span class="top-badge">Cobertura base disponible: 2015–2020</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# BARRA DE HERRAMIENTAS HORIZONTAL COMPACTA
# ══════════════════════════════════════════════════════════════════════════════
c_tool1, c_tool2, c_tool3, c_tool4, c_tool5 = st.columns([2.5, 2.2, 2.5, 1.8, 1.0])

with c_tool1:
    d_input = st.date_input(
        "Periodo",
        value=st.session_state["date_range"],
        min_value=datetime.date(2000, 11, 1),
        max_value=datetime.date.today(),
        label_visibility="collapsed",
        key="toolbar_date_range_global_v4",
    )
    if isinstance(d_input, (tuple, list)) and len(d_input) == 2:
        if d_input != st.session_state["date_range"]:
            st.session_state["date_range"] = tuple(d_input)

with c_tool2:
    quick_region = st.selectbox(
        "Región rápida",
        options=["Seleccionar región rápida..."] + list(REGIONES_PRESET.keys()),
        index=0,
        label_visibility="collapsed",
        key="toolbar_quick_region",
    )
    if quick_region != "Seleccionar región rápida...":
        p_w, p_s, p_e, p_n = REGIONES_PRESET[quick_region]
        st.session_state["active_selection"] = {
            "bbox": (p_w, p_s, p_e, p_n),
            "name": quick_region,
            "source": "Preset",
        }
        st.session_state["map_center"] = [(p_s + p_n) / 2, (p_w + p_e) / 2]
        st.session_state["map_zoom"] = 6 if "Completa" in quick_region else 7
        st.session_state["show_panel"] = True
        st.rerun()

with c_tool3:
    with st.popover("⚙️ Filtros de Sensores"):
        st.markdown("**Instrumentos Satelitales**")
        all_src = sorted(df_raw["source"].cat.categories.tolist())
        st.session_state["sel_sources"] = st.multiselect("Sensores:", options=all_src, default=st.session_state["sel_sources"])
        st.markdown("**Nivel de Confianza**")
        all_cnf = sorted(df_raw["confidence_level"].cat.categories.tolist())
        st.session_state["sel_conf"] = st.multiselect("Confianza:", options=all_cnf, default=st.session_state["sel_conf"])
        st.session_state["show_reference_layer"] = st.checkbox("Mostrar capa base histórica en mapa", value=st.session_state["show_reference_layer"])

with c_tool4:
    if st.session_state["active_selection"]:
        if not st.session_state["show_panel"]:
            if st.button("📊 Ver Estadísticas", width="stretch", type="primary"):
                st.session_state["show_panel"] = True
                st.rerun()
        else:
            if st.button("✕ Ocultar Panel", width="stretch"):
                st.session_state["show_panel"] = False
                st.rerun()
    else:
        st.caption("👈 Dibuje ▢ o elija región")

with c_tool5:
    with st.popover("ℹ️ Info"):
        st.markdown("**Acerca de FireTrackers**")
        st.markdown("""
        Explorador geoespacial de anomalías y tendencias de actividad de fuego basado en datos armonizados de NASA FIRMS (sensores MODIS y VIIRS).

        - **Deduplicación:** Ventana espacio-temporal de 1 km y 3 horas.
        - **Normalización:** Homologación de niveles de confianza y cálculo de FRP.
        - **Cobertura histórica disponible:** 2015–2020 (521.534 registros procesados).
        """)
        if st.session_state["active_selection"]:
            if st.button("Limpiar selección", width="stretch"):
                st.session_state["active_selection"] = None
                st.session_state["show_panel"] = False
                st.rerun()


@st.cache_data(ttl=3600, show_spinner="Consultando detecciones NASA FIRMS...")
def load_nasa_preview(bbox, inicio, fin):
    """Consulta remota una vez por región y fechas; no recalcula al mover el mapa."""
    resultado = request_ingest_preview(bbox, inicio, fin)
    if not resultado.get("success"):
        return resultado, pd.DataFrame()

    tablas = []
    for nombre, ruta in resultado.get("archivos", {}).items():
        csv_path = Path(ruta)
        if not csv_path.exists():
            continue
        try:
            tabla = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            continue
        if tabla.empty:
            continue
        tabla["timestamp"] = pd.to_datetime(tabla["timestamp"], errors="coerce")
        tabla["frp"] = pd.to_numeric(tabla["frp"], errors="coerce")
        tabla["latitude"] = pd.to_numeric(tabla["latitude"], errors="coerce")
        tabla["longitude"] = pd.to_numeric(tabla["longitude"], errors="coerce")
        tabla = tabla.dropna(subset=["timestamp", "latitude", "longitude"])
        tabla["source"] = "modis" if nombre == "modis" else "viirs-snpp" if nombre == "viirs_snpp" else "viirs-noaa20"
        tabla["year"] = tabla["timestamp"].dt.year
        tablas.append(tabla)
    return resultado, pd.concat(tablas, ignore_index=True) if tablas else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner="Armonizando detecciones y calculando estadísticas...")
def load_region_analysis(bbox, inicio, fin):
    """Ejecuta Sprints 1-3 una sola vez por consulta en la sesión/caché."""
    respuesta = request_full_pipeline(bbox, inicio, fin)
    if respuesta.get("status") != "pipeline_completed":
        return respuesta, pd.DataFrame()
    tabla = pd.read_csv(respuesta["csv_path"], low_memory=False)
    if not tabla.empty:
        tabla["timestamp"] = pd.to_datetime(tabla["timestamp"], errors="coerce", utc=True)
        tabla["frp"] = pd.to_numeric(tabla.get("frp"), errors="coerce")
        tabla["year"] = tabla["timestamp"].dt.year
        tabla["month"] = tabla["timestamp"].dt.month
        tabla = tabla.dropna(subset=["timestamp", "latitude", "longitude"])
    return respuesta, tabla


# ══════════════════════════════════════════════════════════════════════════════
# PROCESAMIENTO DE DATOS (Solo si existe selección activa)
# ══════════════════════════════════════════════════════════════════════════════
date_start, date_end = st.session_state["date_range"]
df_active_filtered = pd.DataFrame()
df_active_region = pd.DataFrame()
has_active_selection = st.session_state["active_selection"] is not None
pipeline_can_visualize = True
pipeline_msg = ""
preview_mode = False
preview_states = {}
external_analytics = False
external_summary = {}

if has_active_selection:
    active_bbox = st.session_state["active_selection"]["bbox"]
    w, s, e, n = active_bbox
    es_base = (
        is_bbox_covered_by_colombia(active_bbox)
        and date_start >= FECHA_MIN_COLOMBIA
        and date_end <= FECHA_MAX_COLOMBIA
    )

    if es_base:
        # Se conserva el análisis científico precalculado de Colombia.
        pipe_res = request_pipeline_execution(active_bbox, date_start, date_end)
        pipeline_can_visualize = pipe_res.get("can_visualize", False)
        if pipeline_can_visualize:
            df_active_region = filter_bbox(df_raw, active_bbox)
            if st.session_state["sel_sources"]:
                df_active_region = df_active_region[df_active_region["source"].isin(st.session_state["sel_sources"])]
            if st.session_state["sel_conf"]:
                df_active_region = df_active_region[df_active_region["confidence_level"].isin(st.session_state["sel_conf"])]
            df_active_filtered = filter_dates(df_active_region, date_start, date_end)
    else:
        # Nueva región: pipeline real Sprints 1, 2 y 3.
        respuesta, tabla = load_region_analysis(tuple(active_bbox), date_start, date_end)
        pipeline_can_visualize = bool(respuesta.get("success"))
        pipeline_msg = respuesta.get("message", "")
        external_analytics = respuesta.get("status") == "pipeline_completed"
        external_summary = respuesta.get("analytics_summary", {})
        preview_states = respuesta.get("estado_sensores", {})
        if pipeline_can_visualize:
            df_active_region = tabla
            df_active_filtered = tabla


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT DINÁMICO: MAPA COMPLETO O MAPA + PANEL LATERAL
# ══════════════════════════════════════════════════════════════════════════════
show_side_panel = has_active_selection and st.session_state["show_panel"]

if show_side_panel:
    col_map, col_panel = st.columns([6.8, 5.2], gap="small")
    map_height = 720
else:
    col_map = st.container()
    col_panel = None
    map_height = 720


# ── Renderizado del Mapa en col_map ──────────────────────────────────────────
with col_map:
    # Mensaje prompt cuando no hay selección activa
    if not has_active_selection:
        st.markdown("""
        <div class="prompt-banner">
            <span>🗺️ <strong>Modo Exploración:</strong> Panee y haga zoom libremente. Para analizar actividad de fuego, <strong>dibuje un rectángulo con el icono ▢</strong> en el mapa o elija una región en la barra superior.</span>
        </div>
        """, unsafe_allow_html=True)

    # Construcción del Mapa Folium
    m = folium.Map(
        location=st.session_state["map_center"],
        zoom_start=st.session_state["map_zoom"],
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Esri Light Gray",
    )

    # Capas base opcionales
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satélite (Esri)",
        overlay=False,
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", overlay=False).add_to(m)

    # Herramienta Draw: Solo Rectángulo
    Draw(
        export=False,
        draw_options={
            "polyline": False,
            "polygon": False,
            "circle": False,
            "circlemarker": False,
            "marker": False,
            "rectangle": True,
        },
        edit_options={"edit": False, "remove": True},
    ).add_to(m)

    # Si hay un área activa seleccionada, resaltarla en el mapa
    if has_active_selection:
        w_sel, s_sel, e_sel, n_sel = st.session_state["active_selection"]["bbox"]
        folium.Rectangle(
            bounds=[[s_sel, w_sel], [n_sel, e_sel]],
            color="#D32F2F",
            weight=2.5,
            dash_array="5, 5",
            fill=True,
            fill_color="#FFCDD2",
            fill_opacity=0.12,
            tooltip=f"Zona activa: {st.session_state['active_selection'].get('name', 'Área seleccionada')}",
        ).add_to(m)

        # Cargar puntos de la zona activa
        n_active_pts = len(df_active_filtered)
        if n_active_pts > 0:
            sample_pts = df_active_filtered if n_active_pts <= MAX_MAP_DETECTIONS else df_active_filtered.sample(MAX_MAP_DETECTIONS, random_state=42)
            cluster_locs = sample_pts[["latitude", "longitude"]].values.tolist()
            FastMarkerCluster(
                data=cluster_locs,
                name="Focos en Zona Activa",
                disableClusteringAtZoom=10,
            ).add_to(m)

            # Marcadores individuales para los focos más intensos
            top_focos = sample_pts.nlargest(min(120, len(sample_pts)), "frp")
            fg_focos = folium.FeatureGroup(name="Detecciones Destacadas", show=True)
            for _, pt in top_focos.iterrows():
                is_v = str(pt["source"]).startswith("viirs")
                frp_val = float(pt["frp"]) if not pd.isna(pt["frp"]) else 0.0
                folium.CircleMarker(
                    location=[float(pt["latitude"]), float(pt["longitude"])],
                    radius=max(3.5, min(7.5, 3.5 + frp_val / 60)),
                    color=COLOR_VIIRS if is_v else COLOR_MODIS,
                    fill=True,
                    fill_color=COLOR_VIIRS if is_v else COLOR_MODIS,
                    fill_opacity=0.85,
                    popup=folium.Popup(
                        f"<b>Sensor:</b> {'VIIRS' if is_v else 'MODIS'}<br>"
                        f"<b>Fecha:</b> {str(pt['timestamp'])[:16]}<br>"
                        f"<b>FRP:</b> {frp_val:.1f} MW<br>"
                        f"<b>Confianza:</b> {pt.get('confidence_level', 'N/A')}",
                        max_width=200,
                    ),
                    tooltip=f"{'VIIRS' if is_v else 'MODIS'} | FRP: {frp_val:.1f} MW",
                ).add_to(fg_focos)
            fg_focos.add_to(m)

    elif st.session_state["show_reference_layer"]:
        # Capa de referencia histórica inicial (muestra ligera para contexto geográfico sin saturar)
        ref_sample = df_raw.sample(min(1500, len(df_raw)), random_state=42)
        ref_locs = ref_sample[["latitude", "longitude"]].values.tolist()
        FastMarkerCluster(
            data=ref_locs,
            name="Cobertura de Referencia Histórica",
            disableClusteringAtZoom=9,
        ).add_to(m)

    # Leyenda compacta en esquina
    legend_code = """
    <div style="position:fixed;bottom:20px;left:12px;z-index:999;background:rgba(255,255,255,0.92);
         padding:5px 10px;border-radius:4px;border:1px solid #cbd5e1;font-size:11px;line-height:1.4;">
      <span style="color:#D32F2F;font-weight:bold;">●</span> VIIRS &nbsp;
      <span style="color:#F57C00;font-weight:bold;">●</span> MODIS &nbsp;
      <span style="color:#D32F2F;">⬚</span> Selección
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_code))
    folium.LayerControl(collapsed=True, position="topright").add_to(m)

    # RENDERIZADO MAP-FIRST: Únicamente captura selección y clic (Sin paneo/zoom repetitivo)
    map_events = st_folium(
        m,
        height=map_height,
        use_container_width=True,
        returned_objects=["last_active_drawing", "last_clicked"],
        key="main_fireatlas_map",
    )


# ══════════════════════════════════════════════════════════════════════════════
# GESTIÓN DE EVENTOS DE DIBUJO Y SELECCIÓN
# ══════════════════════════════════════════════════════════════════════════════
drawing = map_events.get("last_active_drawing") if map_events else None
if drawing:
    drawing_id = str(drawing)
    if drawing_id != st.session_state.get("_last_drawing_id"):
        try:
            geom = shape(drawing["geometry"])
            db = geom.bounds
            drawn_bbox = (
                float(round(db[0], 3)),
                float(round(db[1], 3)),
                float(round(db[2], 3)),
                float(round(db[3], 3)),
            )
            is_valid, _ = validate_coordinates(drawn_bbox[1], drawn_bbox[3], drawn_bbox[0], drawn_bbox[2])
            if is_valid:
                st.session_state["active_selection"] = {
                    "bbox": drawn_bbox,
                    "name": "Área Delimitada en Mapa",
                    "source": "Dibujo",
                }
                st.session_state["map_center"] = [
                    (drawn_bbox[1] + drawn_bbox[3]) / 2,
                    (drawn_bbox[0] + drawn_bbox[2]) / 2,
                ]
                st.session_state["show_panel"] = True
                st.session_state["_last_drawing_id"] = drawing_id
                st.rerun()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# PANEL CONTEXTUAL ESTADÍSTICO (Visible SOLO al seleccionar zona)
# ══════════════════════════════════════════════════════════════════════════════
if show_side_panel and col_panel is not None:
    with col_panel:
        st.markdown('<span class="ft-overlay-anchor" aria-hidden="true"></span>', unsafe_allow_html=True)
        sel_info = st.session_state["active_selection"]
        bw, bs, be, bn = sel_info["bbox"]

        # Encabezado del panel con botón de cierre ✕
        c_phead1, c_phead2 = st.columns([5, 1])
        with c_phead1:
            st.markdown(f"""
            <div class="panel-header">
                <div>
                    <h2 class="panel-title">{sel_info.get('name', 'Zona Seleccionada')}</h2>
                    <p class="panel-subtitle">Lat [{bs:.2f}°, {bn:.2f}°] · Lon [{bw:.2f}°, {be:.2f}°]</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c_phead2:
            if st.button("✕", help="Cerrar panel y volver al mapa completo"):
                st.session_state["show_panel"] = False
                st.rerun()

        # Validación de cobertura y datos
        if not pipeline_can_visualize:
            st.info("ℹ️ " + pipeline_msg)
        elif external_analytics:
            st.caption("Datos NASA FIRMS procesados con Sprints 1 → 2 → 3. Son detecciones satelitales, no incendios únicos.")
            if preview_states:
                st.caption("Disponibilidad de sensores: " + ", ".join(f"{k}: {v}" for k, v in preview_states.items()))
            if df_active_filtered.empty:
                st.info("No quedaron detecciones tras los filtros de armonización.")
            else:
                tot = len(df_active_filtered)
                m1, m2, m3 = st.columns(3)
                m1.metric("Detecciones armonizadas", f"{tot:,}")
                counts_sensor = df_active_filtered["source"].value_counts()
                m2.metric("Sensores con registros", str(len(counts_sensor)))
                m3.metric("Días con detecciones", str(df_active_filtered["timestamp"].dt.date.nunique()))

                t_cal, t_anom, t_tend, t_sens, t_exp, t_play, t_zonas = st.tabs([
                    "📊 Calendario", "📈 Anomalías", "📉 Tendencia", "🛰️ Sensores", "💾 Exportar", "▶️ Evolución", "🧭 Zonas"])
                with t_cal:
                    mensual = (df_active_filtered.assign(
                        año=df_active_filtered["timestamp"].dt.year,
                        mes=df_active_filtered["timestamp"].dt.month)
                        .groupby(["año", "mes"]).size().unstack(fill_value=0)
                        .reindex(columns=range(1, 13), fill_value=0))
                    if not mensual.empty:
                        # Los meses no consultados deben quedar como sin cobertura, no ceros.
                        inicio_mes = date_start.year * 12 + date_start.month
                        fin_mes = date_end.year * 12 + date_end.month
                        for ano in mensual.index:
                            for mes in mensual.columns:
                                k = int(ano) * 12 + int(mes)
                                if k < inicio_mes or k > fin_mes:
                                    mensual.loc[ano, mes] = float("nan")
                        fig = go.Figure(go.Heatmap(
                            z=mensual.values, x=list(MESES_ES), y=mensual.index.astype(str),
                            colorscale=HEATMAP_COLORSCALE,
                            hovertemplate="Año %{y}, mes %{x}: %{z} detecciones<extra></extra>"))
                        fig.update_layout(height=245, margin=dict(l=5, r=5, t=10, b=5))
                        st.plotly_chart(fig, width="stretch")
                    st.caption("Solo se representan los meses incluidos en la consulta; el resto no tiene cobertura confirmada.")
                with t_anom:
                    st.info("Anomalías no disponibles: esta consulta no incluye una referencia histórica multianual con cobertura verificada. No se calculan anomalías a partir de pocos días.")
                with t_tend:
                    dias = (df_active_filtered.groupby(df_active_filtered["timestamp"].dt.date)
                            .size().rename("Detecciones").reset_index())
                    dias.columns = ["Fecha", "Detecciones"]
                    fig = px.bar(dias, x="Fecha", y="Detecciones", title="Detecciones diarias")
                    fig.update_layout(height=270, margin=dict(l=5, r=5, t=35, b=15))
                    st.plotly_chart(fig, width="stretch")
                    st.caption("Conteos dentro del intervalo consultado, no evolución anual global.")
                with t_sens:
                    distrib = counts_sensor.rename_axis("Sensor").reset_index(name="Detecciones")
                    fig = px.pie(distrib, names="Sensor", values="Detecciones", hole=0.4)
                    fig.update_layout(height=260, margin=dict(l=5, r=5, t=5, b=5))
                    st.plotly_chart(fig, width="stretch")
                    if "confidence_level" in df_active_filtered.columns:
                        st.dataframe(df_active_filtered["confidence_level"].value_counts().rename("Detecciones"), width="stretch")
                with t_exp:
                    columnas = [c for c in ["timestamp", "source", "latitude", "longitude", "frp",
                                            "confidence", "confidence_level", "type_unverified", "source_raw", "input_file"]
                                if c in df_active_filtered.columns]
                    export = df_active_filtered[columnas]
                    st.dataframe(export.head(50), width="stretch", hide_index=True)
                    st.download_button("📥 Descargar CSV armonizado", data=export.to_csv(index=False).encode("utf-8"),
                                       file_name="detecciones_armonizadas.csv", mime="text/csv")
                    st.caption("Los archivos y resúmenes analíticos de la consulta se conservan en data/queries/.")
                with t_play:
                    render_player(df_active_filtered, (bw, bs, be, bn), date_start, date_end)
                with t_zonas:
                    render_activity_zones(df_active_filtered, (bw, bs, be, bn))
        elif preview_mode:
            st.warning("Vista preliminar NASA FIRMS: detecciones sin armonizar ni deduplicar. No representan las estadísticas finales.")
            st.metric("Detecciones recibidas y filtradas", f"{len(df_active_filtered):,}")
            if preview_states:
                st.caption("Disponibilidad: " + ", ".join(f"{k}: {v}" for k, v in preview_states.items()))
            if df_active_filtered.empty:
                st.info("No hay detecciones para esta región y periodo, o las fuentes no tuvieron cobertura.")
            else:
                columnas = [c for c in ["timestamp", "source", "latitude", "longitude", "frp", "confidence", "type"] if c in df_active_filtered.columns]
                st.dataframe(df_active_filtered[columnas].head(50), width="stretch", hide_index=True)
                st.download_button(
                    "Descargar detecciones preliminares (CSV)",
                    data=df_active_filtered[columnas].to_csv(index=False).encode("utf-8"),
                    file_name="firetrackers_detecciones_sin_armonizar.csv",
                    mime="text/csv",
                )
        elif len(df_active_filtered) == 0:
            st.warning("No se registraron detecciones satelitales en esta área para el periodo seleccionado.")
        else:
            total_pts = len(df_active_filtered)

            # KPIs Compactos en cuadrícula 3x2
            k1, k2, k3 = st.columns(3)
            k1.metric("Detecciones", f"{total_pts:,}")

            ann_counts = df_active_filtered.groupby("year").size()
            k2.metric("Año Pico", str(ann_counts.idxmax()) if not ann_counts.empty else "N/A",
                      f"{ann_counts.max():,} focos" if not ann_counts.empty else "")

            mon_counts = monthly_counts(df_active_filtered).groupby("month_name")["count"].sum()
            k3.metric("Mes Pico", str(mon_counts.idxmax()) if not mon_counts.empty else "N/A",
                      f"{mon_counts.max():,} focos" if not mon_counts.empty else "")

            # Cálculos analíticos para la región
            baseline_yrs = [y for y in [2015, 2016, 2017, 2018, 2019] if y in df_active_region["year"].unique()]
            med_reg = historical_median(df_active_region, baseline_years=baseline_yrs)
            anom_df = compute_anomalies(df_active_filtered, median_df=med_reg)
            cal_df = calendar_matrix(df_active_filtered, all_years=sorted(df_active_filtered["year"].unique()))

            # Pestañas analíticas compactas dentro del panel
            t_cal, t_anom, t_tend, t_sens, t_exp, t_play, t_zonas = st.tabs([
                "📊 Calendario",
                "📈 Anomalías",
                "📉 Tendencia",
                "🛰️ Sensores",
                "💾 Exportar", "▶️ Evolución", "🧭 Zonas"])

            # 1. Calendario Heatmap
            with t_cal:
                st.caption("Frecuencia mensual (Año × Mes) en la zona seleccionada:")
                if not cal_df.empty:
                    fig_h = go.Figure(
                        data=go.Heatmap(
                            z=cal_df.values,
                            x=cal_df.columns.tolist(),
                            y=[str(y) for y in cal_df.index.tolist()],
                            colorscale=HEATMAP_COLORSCALE,
                            hovertemplate="<b>Año %{y} — %{x}</b><br>Focos: %{z:,.0f}<extra></extra>",
                            colorbar=dict(thickness=10, len=0.8),
                            zmin=0,
                        )
                    )
                    fig_h.update_layout(
                        height=max(220, len(cal_df) * 36 + 40),
                        margin=dict(l=10, r=10, t=10, b=10),
                        xaxis=dict(side="top", tickfont=dict(size=10)),
                        yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
                    )
                    st.plotly_chart(fig_h, width="stretch")

            # 2. Anomalías vs Mediana
            with t_anom:
                st.caption(f"Comparación vs mediana histórica (años {baseline_yrs}):")
                if not anom_df.empty and not anom_df["anomaly_ratio"].dropna().empty:
                    fig_a = px.bar(
                        anom_df,
                        x="year_month",
                        y="anomaly_ratio",
                        color="anomaly_label",
                        color_discrete_map=ANOMALY_COLOR_MAP,
                        labels={"year_month": "Periodo", "anomaly_ratio": "Ratio vs Mediana"},
                        hover_data={"count": ":,.0f", "median_historical": ":,.1f", "display_ratio": True, "anomaly_ratio": False},
                    )
                    fig_a.add_hline(y=1.0, line_dash="dash", line_color="#475569", annotation_text="Mediana (1.0×)")
                    fig_a.add_hline(y=ANOMALY_THRESHOLD_HIGH, line_dash="dot", line_color="#D32F2F")
                    fig_a.update_layout(
                        height=260,
                        margin=dict(l=10, r=10, t=20, b=40),
                        xaxis_tickangle=-45,
                        showlegend=False,
                    )
                    st.plotly_chart(fig_a, width="stretch")

            # 3. Tendencia y Estacionalidad
            with t_tend:
                col_tr1, col_tr2 = st.columns(2)
                with col_tr1:
                    st.caption("Serie anual de incendios:")
                    ann_df = ann_counts.reset_index()
                    ann_df.columns = ["Año", "Focos"]
                    fig_an = px.bar(ann_df, x="Año", y="Focos", color_discrete_sequence=["#D32F2F"])
                    fig_an.update_layout(height=210, margin=dict(l=10, r=10, t=10, b=20))
                    st.plotly_chart(fig_an, width="stretch")
                with col_tr2:
                    st.caption("Perfil estacional (media/mes):")
                    seas = monthly_counts(df_active_filtered).groupby(["month", "month_name"])["count"].mean().reset_index().sort_values("month")
                    fig_s = px.line(seas, x="month_name", y="count", markers=True, color_discrete_sequence=["#F57C00"])
                    fig_s.update_layout(height=210, margin=dict(l=10, r=10, t=10, b=20))
                    st.plotly_chart(fig_s, width="stretch")

            # 4. Sensores
            with t_sens:
                col_sn1, col_sn2 = st.columns(2)
                with col_sn1:
                    st.caption("Proporción por sensor:")
                    pie_df = df_active_filtered["source"].value_counts().reset_index()
                    pie_df.columns = ["Sensor", "Total"]
                    fig_p = px.pie(
                        pie_df, names="Sensor", values="Total",
                        color="Sensor",
                        color_discrete_map={"viirs-snpp": COLOR_VIIRS, "modis": COLOR_MODIS},
                        hole=0.45,
                    )
                    fig_p.update_layout(height=210, margin=dict(l=5, r=5, t=10, b=10))
                    st.plotly_chart(fig_p, width="stretch")
                with col_sn2:
                    st.caption("Nivel de confianza:")
                    cnf_df = df_active_filtered["confidence_level"].value_counts().reset_index()
                    cnf_df.columns = ["Confianza", "Total"]
                    fig_cf = px.bar(cnf_df, x="Confianza", y="Total", color_discrete_sequence=["#334155"])
                    fig_cf.update_layout(height=210, margin=dict(l=5, r=5, t=10, b=10))
                    st.plotly_chart(fig_cf, width="stretch")

            # 5. Exportar CSV
            with t_exp:
                st.caption("Descargar registros de la zona activa:")
                exp_cols = [
                    "latitude", "longitude", "acq_date", "acq_time", "timestamp",
                    "satellite", "instrument", "source", "confidence_level",
                    "frp", "daynight", "year",
                ]
                safe_exp = [c for c in exp_cols if c in df_active_filtered.columns]
                df_to_dl = sanitize_for_csv(df_active_filtered, columns=safe_exp)
                st.dataframe(df_to_dl.head(5), width="stretch", hide_index=True)

                csv_data = df_to_dl.to_csv(index=False, encoding="utf-8").encode("utf-8")
                fname = f"firetrackers_{bs:.2f}N_{bn:.2f}N_{bw:.2f}E_{be:.2f}E.csv"
                st.download_button(
                    label=f"📥 Descargar CSV ({len(df_to_dl):,} registros)",
                    data=csv_data,
                    file_name=fname,
                    mime="text/csv",
                    type="primary",
                    width="stretch",
                )

            # Animación de detecciones de Colombia precalculadas.
            with t_play:
                render_player(df_active_filtered, (bw, bs, be, bn), date_start, date_end)
            with t_zonas:
                render_activity_zones(df_active_filtered, (bw, bs, be, bn))
