"""
Aplicación Web MVP - GEOVISOR INTERACTIVO CENTRADO EN EL MAPA
NASA Space Apps Challenge 2026 - Harmonization of MODIS and VIIRS Hot Spots
Sprint 4: El mapa es el controlador interactivo principal de la aplicación.
La selección de área, paneo/zoom y clics en detecciones disparan automáticamente
los análisis históricos, calendario, anomalías y orquestación del pipeline.
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

# Asegurar importación de módulos hermanos
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

from pipeline_adapter import (
    check_pipeline_modules,
    request_pipeline_execution,
    get_required_contracts_doc,
    is_bbox_covered_by_colombia,
    BBOX_COLOMBIA_COBERTURA,
)

# --- Rutas de datos ---
DATA_DIR = BASE_DIR / "data"
HARMONIZED_DIR = DATA_DIR / "harmonized"
ANALYTICS_DIR = DATA_DIR / "analytics"

# --- Paletas de color accesibles y contrastantes ---
HEATMAP_COLORSCALE = "YlOrRd"
ANOMALY_COLOR_MAP = {
    "alta": "#D32F2F",                # Rojo accesible
    "alta (mediana 0)": "#E65100",     # Naranja oscuro
    "normal": "#2E7D32",              # Verde bosque
    "baja": "#1976D2",                # Azul zafiro
    "sin_referencia": "#757575"       # Gris neutro
}

REGIONES_PRESET = {
    "Toda Colombia (Nacional)": (-81.71, -4.23, -66.90, 12.56),
    "Región Caribe (Norte)": (-77.50, 7.80, -71.00, 12.56),
    "Región Andina (Centro)": (-77.20, 1.50, -72.50, 8.50),
    "Región Amazonía (Sur)": (-75.50, -4.23, -69.50, 1.50),
    "Región Orinoquía (Oriente)": (-73.50, 1.50, -67.00, 6.50),
    "Región Pacífica (Occidente)": (-79.20, 1.00, -76.20, 7.50)
}

# --- Configuración de página amplia para Geovisor ---
st.set_page_config(
    page_title="Geovisor de Incendios | NASA FIRMS",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed"  # Colapsado por defecto para dar protagonismo al mapa
)

# Inyección de estilos CSS para un geovisor limpio y profesional
st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .stMetric { background: #f8f9fa; padding: 10px 15px; border-radius: 8px; border-left: 4px solid #D32F2F; }
    .leaflet-container { border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }
</style>
""", unsafe_allow_html=True)


# --- Carga optimizada de dataset nacional ---
@st.cache_data(show_spinner=False)
def load_harmonized_data():
    csv_path = HARMONIZED_DIR / "incendios_armonizados.csv"
    if not csv_path.exists():
        return None, "No se encontró data/harmonized/incendios_armonizados.csv"

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
        "priority": "int8"
    }

    try:
        df = pd.read_csv(csv_path, dtype=dtypes, parse_dates=["timestamp"])
        df["month"] = df["timestamp"].dt.month.astype("int8")
        df["month_name"] = df["month"].map(MESES_ES).astype("category")
        df["year_month"] = df["timestamp"].dt.to_period("M").astype(str)
        return df, ""
    except Exception:
        return None, "Error interno al leer archivo de incendios armonizados."


df_raw, load_error = load_harmonized_data()
if df_raw is None or df_raw.empty:
    st.error(f"⚠️ {load_error}")
    st.stop()

ds_lat_min, ds_lat_max = float(df_raw["latitude"].min()), float(df_raw["latitude"].max())
ds_lon_min, ds_lon_max = float(df_raw["longitude"].min()), float(df_raw["longitude"].max())
ds_min_date = df_raw["timestamp"].min().date()
ds_max_date = df_raw["timestamp"].max().date()

# ==========================================
# Inicialización de Estado de Sesión (Session State)
# ==========================================
if "active_bbox" not in st.session_state:
    st.session_state["active_bbox"] = (ds_lon_min, ds_lat_min, ds_lon_max, ds_lat_max)
if "map_center" not in st.session_state:
    st.session_state["map_center"] = [4.5, -73.5]
if "map_zoom" not in st.session_state:
    st.session_state["map_zoom"] = 6
if "selected_object" not in st.session_state:
    st.session_state["selected_object"] = None
if "sync_with_viewport" not in st.session_state:
    st.session_state["sync_with_viewport"] = False
if "date_range" not in st.session_state:
    st.session_state["date_range"] = (ds_min_date, ds_max_date)
if "trigger_source" not in st.session_state:
    st.session_state["trigger_source"] = "Inicial"

# ==========================================
# Encabezado del Geovisor
# ==========================================
col_title, col_ctrls = st.columns([5, 4])
with col_title:
    st.markdown("### 🔥 Geovisor Interactivo de Incendios | NASA FIRMS (MODIS + VIIRS)")
    st.caption("🌍 **El mapa es el controlador:** Dibuje un rectángulo con el icono ▢ o ajuste la vista para filtrar automáticamente las estadísticas.")

with col_ctrls:
    # Controles compactos superiores
    c_preset, c_dates = st.columns([2, 2])
    with c_preset:
        sel_preset = st.selectbox(
            "Región Rápida:",
            options=["Seleccionar preset..."] + list(REGIONES_PRESET.keys()),
            index=0,
            label_visibility="collapsed"
        )
        if sel_preset != "Seleccionar preset...":
            p_w, p_s, p_e, p_n = REGIONES_PRESET[sel_preset]
            st.session_state["active_bbox"] = (p_w, p_s, p_e, p_n)
            st.session_state["map_center"] = [(p_s + p_n) / 2, (p_w + p_e) / 2]
            st.session_state["map_zoom"] = 6 if sel_preset.startswith("Toda") else 7
            st.session_state["trigger_source"] = f"Preset: {sel_preset}"

    with c_dates:
        # Selector compacto de fechas
        d_input = st.date_input(
            "Rango de Fechas:",
            value=st.session_state["date_range"],
            min_value=ds_min_date,
            max_value=ds_max_date,
            label_visibility="collapsed"
        )
        if isinstance(d_input, (tuple, list)) and len(d_input) == 2:
            st.session_state["date_range"] = d_input

date_start, date_end = st.session_state["date_range"]
is_dt_valid, dt_msg = validate_dates(date_start, date_end, ds_min_date, ds_max_date)
if not is_dt_valid:
    st.error(f"⚠️ {dt_msg}")
    st.stop()

# ==========================================
# Barra Lateral (Configuración y Filtros Opcionales)
# ==========================================
with st.sidebar:
    st.header("⚙️ Ajustes del Geovisor")
    
    st.subheader("🛰️ Filtros por Sensor y Confianza")
    avail_sources = sorted(df_raw["source"].unique())
    sel_sources = st.multiselect("Sensores satelitales:", options=avail_sources, default=avail_sources)
    
    avail_conf = sorted(df_raw["confidence_level"].unique())
    sel_conf = st.multiselect("Nivel de confianza:", options=avail_conf, default=avail_conf)
    
    st.divider()
    st.subheader("🧭 Sincronización Automática")
    sync_viewport = st.checkbox(
        "Sincronizar con paneo y zoom",
        value=st.session_state["sync_with_viewport"],
        help="Si está activo, al desplazar o hacer zoom en el mapa, las estadísticas se recalculan para la ventana visible."
    )
    st.session_state["sync_with_viewport"] = sync_viewport

    st.divider()
    st.subheader("📐 Coordenadas del Bounding Box Activo")
    cur_w, cur_s, cur_e, cur_n = st.session_state["active_bbox"]
    c_lat1, c_lat2 = st.columns(2)
    new_s = c_lat1.number_input("Lat Mín", value=float(round(cur_s, 2)), format="%.2f")
    new_n = c_lat2.number_input("Lat Máx", value=float(round(cur_n, 2)), format="%.2f")
    c_lon1, c_lon2 = st.columns(2)
    new_w = c_lon1.number_input("Lon Mín", value=float(round(cur_w, 2)), format="%.2f")
    new_e = c_lon2.number_input("Lon Máx", value=float(round(cur_e, 2)), format="%.2f")

    if (new_w != cur_w or new_s != cur_s or new_e != cur_e or new_n != cur_n):
        val_c, msg_c = validate_coordinates(new_s, new_n, new_w, new_e)
        if val_c:
            st.session_state["active_bbox"] = (new_w, new_s, new_e, new_n)
            st.session_state["trigger_source"] = "Entrada Manual"
            st.rerun()

    if st.button("🔄 Restablecer a Toda Colombia", use_container_width=True):
        st.session_state["active_bbox"] = (ds_lon_min, ds_lat_min, ds_lon_max, ds_lat_max)
        st.session_state["map_center"] = [4.5, -73.5]
        st.session_state["map_zoom"] = 6
        st.session_state["trigger_source"] = "Restablecer"
        st.rerun()

# ==========================================
# SECCIÓN 1: EL GEOVISOR COMO PROTAGONISTA
# ==========================================
active_w, active_s, active_e, active_n = st.session_state["active_bbox"]

# Filtrar datos de la región para graficar en el mapa
df_active_region = filter_bbox(df_raw, st.session_state["active_bbox"])
if sel_sources:
    df_active_region = df_active_region[df_active_region["source"].isin(sel_sources)]
if sel_conf:
    df_active_region = df_active_region[df_active_region["confidence_level"].isin(sel_conf)]
df_active_filtered = filter_dates(df_active_region, date_start, date_end)

# Construir mapa interactivo Folium
m = folium.Map(
    location=st.session_state["map_center"],
    zoom_start=st.session_state["map_zoom"],
    tiles="OpenStreetMap"
)

# Herramienta Draw para dibujo de rectángulos
Draw(
    export=False,
    draw_options={
        "polyline": False,
        "polygon": False,
        "circle": False,
        "circlemarker": False,
        "marker": False,
        "rectangle": True
    },
    edit_options={"edit": True, "remove": True}
).add_to(m)

# Resaltar Bounding Box activo
folium.Rectangle(
    bounds=[[active_s, active_w], [active_n, active_e]],
    color="#D32F2F",
    weight=3,
    fill=True,
    fill_color="#FFCDD2",
    fill_opacity=0.2,
    popup=f"Área activa de consulta: {len(df_active_filtered):,} focos",
    tooltip="Área de consulta activa"
).add_to(m)

# Capa de puntos reales con clustering eficiente para rendimiento
# Muestra hasta 3.000 puntos interactivos en la zona activa sin degradar el navegador
MAX_MAP_DETECTIONS = 3000
if len(df_active_filtered) > 0:
    pts_sample = df_active_filtered if len(df_active_filtered) <= MAX_MAP_DETECTIONS else df_active_filtered.sample(MAX_MAP_DETECTIONS, random_state=42)
    
    # Preparar datos de FastMarkerCluster: lista de [lat, lon]
    cluster_data = pts_sample[["latitude", "longitude"]].values.tolist()
    FastMarkerCluster(data=cluster_data, name="Focos Térmicos (Cluster)").add_to(m)

    # Añadir marcadores individuales detallados para los 200 focos más intensos (FRP alto)
    top_focos = pts_sample.nlargest(min(200, len(pts_sample)), "frp")
    fg_top = folium.FeatureGroup(name="Detecciones destacadas (FRP Alto)")
    for _, pt in top_focos.iterrows():
        is_viirs = str(pt["source"]) == "viirs-snpp"
        col = "#D32F2F" if is_viirs else "#FB8C00"
        folium.CircleMarker(
            location=[float(pt["latitude"]), float(pt["longitude"])],
            radius=4,
            color=col,
            fill=True,
            fill_color=col,
            fill_opacity=0.8,
            popup=folium.Popup(
                f"<b>Detección Satelital:</b> {pt['source']}<br>"
                f"<b>Fecha:</b> {pt['timestamp']}<br>"
                f"<b>Confianza:</b> {pt.get('confidence_level', 'N/A')}<br>"
                f"<b>FRP:</b> {pt.get('frp', 0.0)} MW<br>"
                f"<b>Satélite:</b> {pt.get('satellite', 'N/A')} ({pt.get('instrument', 'N/A')})",
                max_width=250
            ),
            tooltip=f"{pt['source']} | FRP: {pt.get('frp', 0.0)} MW"
        ).add_to(fg_top)
    fg_top.add_to(m)

folium.LayerControl(collapsed=True).add_to(m)

# RENDERIZADO DEL MAPA PRINCIPAL
map_events = st_folium(
    m,
    height=480,
    use_container_width=True,
    returned_objects=["last_active_drawing", "bounds", "last_clicked"],
    key="main_geovisor"
)

# ==========================================
# GESTIÓN AUTOMÁTICA DE EVENTOS DEL MAPA
# ==========================================
map_needs_rerun = False

# Evento A: Dibujo de Rectángulo (Prioridad 1)
if map_events and map_events.get("last_active_drawing"):
    try:
        drawing_geom = shape(map_events["last_active_drawing"]["geometry"])
        db = drawing_geom.bounds # minx, miny, maxx, maxy -> lon_min, lat_min, lon_max, lat_max
        new_bbox = (float(round(db[0], 2)), float(round(db[1], 2)), float(round(db[2], 2)), float(round(db[3], 2)))
        
        # Comprobar si es un dibujo nuevo diferente al actual
        if new_bbox != st.session_state["active_bbox"]:
            is_valid_d, _ = validate_coordinates(new_bbox[1], new_bbox[3], new_bbox[0], new_bbox[2])
            if is_valid_d:
                st.session_state["active_bbox"] = new_bbox
                st.session_state["map_center"] = [(new_bbox[1] + new_bbox[3]) / 2, (new_bbox[0] + new_bbox[2]) / 2]
                st.session_state["trigger_source"] = "Rectángulo Dibujado"
                map_needs_rerun = True
    except Exception:
        pass

# Evento B: Paneo y Zoom (Si la sincronización está activa y no hay dibujo reciente)
elif st.session_state["sync_with_viewport"] and map_events and map_events.get("bounds"):
    b_view = map_events["bounds"]
    # bounds tiene _southWest y _northEast
    if "_southWest" in b_view and "_northEast" in b_view:
        view_s = float(round(b_view["_southWest"]["lat"], 2))
        view_w = float(round(b_view["_southWest"]["lng"], 2))
        view_n = float(round(b_view["_northEast"]["lat"], 2))
        view_e = float(round(b_view["_northEast"]["lng"], 2))
        
        # Aplicar umbral de cambio para no disparar con micromovimientos (> 0.25 grados)
        cur_w, cur_s, cur_e, cur_n = st.session_state["active_bbox"]
        diff = max(abs(view_w - cur_w), abs(view_s - cur_s), abs(view_e - cur_e), abs(view_n - cur_n))
        if diff >= 0.25:
            is_valid_v, _ = validate_coordinates(view_s, view_n, view_w, view_e)
            if is_valid_v:
                st.session_state["active_bbox"] = (view_w, view_s, view_e, view_n)
                st.session_state["map_center"] = [(view_s + view_n) / 2, (view_w + view_e) / 2]
                st.session_state["trigger_source"] = "Paneo / Zoom de Ventana"
                map_needs_rerun = True

# Evento C: Clic en el Mapa
if map_events and map_events.get("last_clicked"):
    st.session_state["selected_object"] = map_events["last_clicked"]

if map_needs_rerun:
    st.rerun()

# ==========================================
# BARRA DE ESTADO DE LA REGIÓN ACTIVA
# ==========================================
c_info1, c_info2, c_info3 = st.columns([4, 4, 3])
with c_info1:
    st.markdown(
        f"📍 **Región Activa:** `Lat: [{active_s:.2f}°, {active_n:.2f}°]` | `Lon: [{active_w:.2f}°, {active_e:.2f}°]`  \n"
        f"*(Origen: {st.session_state['trigger_source']})*"
    )
with c_info2:
    st.markdown(
        f"📅 **Periodo:** `{date_start}` al `{date_end}` ({(date_end - date_start).days + 1:,} días)  \n"
        f"*(Comparación contra mediana histórica territorial)*"
    )
with c_info3:
    if st.session_state.get("selected_object"):
        clat = st.session_state["selected_object"]["lat"]
        clon = st.session_state["selected_object"]["lng"]
        st.markdown(f"🎯 **Último Clic:** `{clat:.4f}°, {clon:.4f}°`")
    else:
        st.markdown("🎯 *Haga clic en un foco para ver su detalle.*")

st.divider()

# ==========================================
# SECCIÓN 2: ORQUESTADOR DEL PIPELINE
# ==========================================
pipeline_res = request_pipeline_execution(st.session_state["active_bbox"], date_start, date_end)

if not pipeline_res["can_visualize"]:
    st.warning("⚠️ **Zona Geográfica fuera de Cobertura Nacional de Colombia**")
    st.markdown(
        f"El área seleccionada `[{active_s:.2f}°, {active_n:.2f}° N, {active_w:.2f}°, {active_e:.2f}° O]` "
        f"requiere ejecutar el pipeline completo de procesamiento: `ingest.py` → `harmonize.py` → `analytics.py`."
    )
    st.error(
        f"**Estado del Pipeline:** {pipeline_res['message']}  \n"
        f"Módulos pendientes de código de los compañeros: `{', '.join(pipeline_res['pending_modules'])}`."
    )
    if st.button("⬅️ Regresar a Cobertura Colombia"):
        st.session_state["active_bbox"] = (ds_lon_min, ds_lat_min, ds_lon_max, ds_lat_max)
        st.rerun()
    st.stop()

# ==========================================
# SECCIÓN 3: PANEL DE ESTADÍSTICAS DINÁMICAS (PANELES INFERIORES)
# ==========================================
if len(df_active_filtered) == 0:
    st.warning(
        "⚠️ **Sin detecciones:** No hay registros satelitales en las coordenadas y periodo seleccionados.  \n"
        "Desplace el mapa o amplíe el rectángulo para explorar áreas activas."
    )
    st.stop()

# Cálculos dinámicos en memoria
baseline_yrs = [y for y in [2015, 2016, 2017, 2018, 2019] if y in df_active_region["year"].unique()]
med_regional = historical_median(df_active_region, baseline_years=baseline_yrs)
anom_dynamic = compute_anomalies(df_active_filtered, median_df=med_regional)
cal_matrix = calendar_matrix(df_active_filtered, all_years=sorted(df_active_filtered["year"].unique()))

# KPIs Principales
k1, k2, k3, k4 = st.columns(4)
total_focos = len(df_active_filtered)
pct_nat = (total_focos / len(df_raw)) * 100
k1.metric("Focos en Zona Activa", f"{total_focos:,}", f"{pct_nat:.1f}% del total nacional")

annual_counts = df_active_filtered.groupby("year").size()
if not annual_counts.empty:
    k2.metric("Año Pico", f"{annual_counts.idxmax()}", f"{annual_counts.max():,} focos")
else:
    k2.metric("Año Pico", "N/A")

month_counts = monthly_counts(df_active_filtered).groupby("month_name")["count"].sum()
if not month_counts.empty:
    k3.metric("Mes de Mayor Actividad", f"{month_counts.idxmax()}", f"{month_counts.max():,} focos")
else:
    k3.metric("Mes de Mayor Actividad", "N/A")

valid_anom = anom_dynamic.dropna(subset=["anomaly_ratio"])
if not valid_anom.empty:
    last_anom = valid_anom.iloc[-1]
    k4.metric(
        f"Última Anomalía ({last_anom['year_month']})",
        last_anom["display_ratio"],
        delta=f"Estado: {last_anom['anomaly_label']}",
        delta_color="inverse" if "alta" in last_anom["anomaly_label"] else "normal"
    )
else:
    k4.metric("Última Anomalía", "Sin ref. histórica")

# PESTAÑAS ANALÍTICAS SIN SALIR DEL GEOVISOR
tab_cal, tab_anom, tab_hist, tab_exp, tab_pipe = st.tabs([
    "📊 Calendario Anual (Heatmap)",
    "📈 Anomalías vs. Mediana Histórica",
    "📉 Distribución y Sensores",
    "💾 Exportar CSV Filtrado",
    "🔗 Integración de Sprints"
])

# 1. Calendario Heatmap
with tab_cal:
    st.subheader("📊 Calendario Histórico de Actividad de Fuego en la Región Activa")
    st.caption("Frecuencia mensual de detecciones en cuadrícula año × mes. Calculado en tiempo real según la zona delimitada.")
    if not cal_matrix.empty:
        fig_heat = go.Figure(
            data=go.Heatmap(
                z=cal_matrix.values,
                x=cal_matrix.columns.tolist(),
                y=[f"Año {y}" for y in cal_matrix.index.tolist()],
                colorscale=HEATMAP_COLORSCALE,
                hovertemplate="<b>%{y} — %{x}</b><br>Detecciones: %{z:,.0f} focos<extra></extra>",
                colorbar=dict(title="Focos", thickness=15, len=0.85)
            )
        )
        fig_heat.update_layout(
            height=max(280, len(cal_matrix) * 40),
            margin=dict(l=30, r=30, t=40, b=30),
            xaxis=dict(title="Mes", side="top"),
            yaxis=dict(title="Año", autorange="reversed")
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# 2. Anomalías vs Mediana
with tab_anom:
    st.subheader("📈 Anomalías Mensuales frente a la Mediana Histórica Territorial")
    st.caption(f"Línea base calculada con los años {baseline_yrs} para exactamente la misma región seleccionada en el mapa.")
    
    col_u1, col_u2, col_u3 = st.columns(3)
    col_u1.markdown(f"🔴 **Alta:** Ratio ≥ `{ANOMALY_THRESHOLD_HIGH}x` (≥50% sobre mediana)")
    col_u2.markdown(f"🟢 **Normal:** Ratio entre `{ANOMALY_THRESHOLD_LOW}x` y `{ANOMALY_THRESHOLD_HIGH}x`")
    col_u3.markdown(f"🔵 **Baja:** Ratio ≤ `{ANOMALY_THRESHOLD_LOW}x` (≤50% de mediana)")

    if not anom_dynamic.empty and not anom_dynamic["anomaly_ratio"].dropna().empty:
        fig_bar = px.bar(
            anom_dynamic,
            x="year_month",
            y="anomaly_ratio",
            color="anomaly_label",
            color_discrete_map=ANOMALY_COLOR_MAP,
            labels={"year_month": "Periodo", "anomaly_ratio": "Ratio vs. Mediana", "anomaly_label": "Clasificación"},
            hover_data={"count": ":,.0f", "median_historical": ":,.1f", "display_ratio": True, "anomaly_ratio": False}
        )
        fig_bar.add_hline(y=1.0, line_dash="dash", line_color="#333", annotation_text="Mediana Histórica (1.0x)")
        fig_bar.update_layout(
            height=340,
            margin=dict(l=30, r=30, t=30, b=50),
            xaxis_tickangle=-45,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# 3. Distribución Temporal y Sensores
with tab_hist:
    st.subheader("📉 Análisis de Sensores y Frecuencia Temporal")
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.markdown("**Comparativa de Sensores Satelitales en la Zona:**")
        fig_pie = px.pie(
            df_active_filtered,
            names="source",
            title="Proporción MODIS vs. VIIRS S-NPP",
            color="source",
            color_discrete_map={"viirs-snpp": "#D32F2F", "modis": "#FB8C00"},
            hole=0.4
        )
        fig_pie.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_h2:
        st.markdown("**Niveles de Confianza Armonizados:**")
        fig_conf = px.bar(
            df_active_filtered["confidence_level"].value_counts().reset_index(),
            x="confidence_level",
            y="count",
            color="confidence_level",
            color_discrete_map={"nominal": "#2E7D32", "high": "#1565C0"},
            labels={"confidence_level": "Nivel de Confianza", "count": "Detecciones"}
        )
        fig_conf.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
        st.plotly_chart(fig_conf, use_container_width=True)

# 4. Exportación CSV
with tab_exp:
    st.subheader("💾 Exportación de la Selección Territorial")
    st.caption("Descarga únicamente los registros que coinciden con los límites del geovisor y fechas activas.")
    
    export_cols = ["latitude", "longitude", "acq_date", "acq_time", "timestamp", "satellite", "instrument", "source", "confidence_level", "frp", "daynight"]
    safe_cols = [c for c in export_cols if c in df_active_filtered.columns]
    df_exp = sanitize_for_csv(df_active_filtered, columns=safe_cols)
    
    st.dataframe(df_exp.head(10), use_container_width=True, hide_index=True)
    
    csv_bytes = df_exp.to_csv(index=False).encode("utf-8")
    fname = f"geovisor_focos_{active_s:.2f}_{active_n:.2f}_{active_w:.2f}_{active_e:.2f}_{date_start}_{date_end}.csv"
    st.download_button(
        f"📥 Descargar CSV ({len(df_exp):,} registros, ~{len(csv_bytes)/1024:.1f} KB)",
        data=csv_bytes,
        file_name=fname,
        mime="text/csv",
        type="primary"
    )

# 5. Integración de Sprints
with tab_pipe:
    st.subheader("🔗 Orquestación de Sprints (1 → 2 → 3 → 4)")
    st.markdown("Verificación técnica de los módulos en `src/` conectados con el geovisor:")
    mod_st = check_pipeline_modules()
    st_table = []
    for k, v in mod_st.items():
        st_table.append({
            "Módulo": v["expected_file"],
            "Ruta": str(v["path"].name),
            "Tamaño": f"{v['size_bytes']} bytes",
            "Estado": "🟢 Operativo" if v["has_required_interface"] else ("🟡 Archivo Vacío (0 B)" if v["size_bytes"] == 0 else "🔴 Sin interfaz BBox"),
            "Diagnóstico": v["details"]
        })
    st.dataframe(pd.DataFrame(st_table), use_container_width=True, hide_index=True)

    with st.expander("Ver Contratos Técnicos para Ingesta y Armonización Dinámica"):
        st.json(get_required_contracts_doc())

st.divider()
st.caption(
    "🛰️ NASA Space Apps Challenge 2026 | Harmonization of MODIS & VIIRS Hot Spots | "
    "Sprint 4: Geovisor interactivo de respuesta en tiempo real. Cobertura: Colombia 2015–2020."
)
