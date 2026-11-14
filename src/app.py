"""
Aplicación Web MVP - Calendario Armonizado de Actividad de Incendios
NASA Space Apps Challenge 2026 - Harmonization of MODIS and VIIRS Hot Spots
Sprint 4: Interfaz de usuario interactiva, analítica geoespacial, exportación segura
y orquestador de integración con los Sprints 1, 2 y 3.
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
from folium.plugins import Draw
from streamlit_folium import st_folium
from shapely.geometry import shape

# Asegurar que el directorio src esté en sys.path
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

# --- Configuración de rutas de datos ---
DATA_DIR = BASE_DIR / "data"
HARMONIZED_DIR = DATA_DIR / "harmonized"
ANALYTICS_DIR = DATA_DIR / "analytics"

# --- Paleta visual accesible ---
HEATMAP_COLORSCALE = "YlOrRd"
ANOMALY_COLOR_MAP = {
    "alta": "#D32F2F",                # Rojo cálido accesible
    "alta (mediana 0)": "#E65100",     # Naranja oscuro
    "normal": "#2E7D32",              # Verde bosque
    "baja": "#1976D2",                # Azul zafiro
    "sin_referencia": "#757575"       # Gris neutro
}

# Regiones geográficas predefinidas de Colombia (w, s, e, n)
REGIONES_PRESET = {
    "Toda Colombia (Cobertura Completa)": (-81.71, -4.23, -66.90, 12.56),
    "Región Caribe (Norte)": (-77.50, 7.80, -71.00, 12.56),
    "Región Andina (Centro)": (-77.20, 1.50, -72.50, 8.50),
    "Región Amazonía (Sur)": (-75.50, -4.23, -69.50, 1.50),
    "Región Orinoquía (Oriente)": (-73.50, 1.50, -67.00, 6.50),
    "Región Pacífica (Occidente)": (-79.20, 1.00, -76.20, 7.50)
}

# --- Configuración de página Streamlit ---
st.set_page_config(
    page_title="Calendario de Incendios | NASA FIRMS",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Carga de datos optimizada en RAM ---
@st.cache_data(show_spinner=False)
def load_harmonized_data():
    """
    Carga el dataset armonizado nacional optimizando tipos de datos.
    """
    csv_path = HARMONIZED_DIR / "incendios_armonizados.csv"
    if not csv_path.exists():
        return None, "No se encontró el archivo de datos en: data/harmonized/incendios_armonizados.csv"

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
        return None, "Error interno al leer el archivo de datos armonizados."


@st.cache_data(show_spinner=False)
def load_sprint3_analytics():
    """Carga resultados precalculados de Sprint 3 si están disponibles."""
    results = {}
    try:
        anom_path = ANALYTICS_DIR / "anomalies.csv"
        if anom_path.exists():
            results["anomalies"] = pd.read_csv(anom_path)
        med_path = ANALYTICS_DIR / "historical_median.csv"
        if med_path.exists():
            results["historical_median"] = pd.read_csv(med_path)
    except Exception:
        pass
    return results


# ==========================================
# Carga inicial y validación de dataset
# ==========================================
df_raw, load_error = load_harmonized_data()

if df_raw is None or df_raw.empty:
    st.error(f"⚠️ {load_error or 'No hay datos disponibles.'}")
    st.info("Verifique que data/harmonized/incendios_armonizados.csv esté presente.")
    st.stop()

# Límites del dataset nacional
ds_lat_min, ds_lat_max = float(df_raw["latitude"].min()), float(df_raw["latitude"].max())
ds_lon_min, ds_lon_max = float(df_raw["longitude"].min()), float(df_raw["longitude"].max())
ds_min_date = df_raw["timestamp"].min().date()
ds_max_date = df_raw["timestamp"].max().date()

# ==========================================
# Gestión de Estado de Filtros (Session State)
# ==========================================
if "lat_min" not in st.session_state:
    st.session_state["lat_min"] = ds_lat_min
if "lat_max" not in st.session_state:
    st.session_state["lat_max"] = ds_lat_max
if "lon_min" not in st.session_state:
    st.session_state["lon_min"] = ds_lon_min
if "lon_max" not in st.session_state:
    st.session_state["lon_max"] = ds_lon_max

# ==========================================
# Barra lateral: Configuración y Filtros
# ==========================================
st.sidebar.title("🔥 Control de Consulta")
st.sidebar.caption("Filtros espaciotemporales y Pipeline de Procesamiento")

# Estado de los módulos del Pipeline
modules_status = check_pipeline_modules()

with st.sidebar.expander("🔗 Estado del Pipeline (Sprints 1–3)", expanded=False):
    s1_ok = modules_status["ingest"]["status"] == "present" and modules_status["ingest"]["has_required_interface"]
    s2_ok = modules_status["harmonize"]["status"] == "present" and modules_status["harmonize"]["has_required_interface"]
    s3_ok = modules_status["analytics"]["status"] == "present" and modules_status["analytics"]["has_required_interface"]

    st.markdown(f"**Sprint 1 (Ingesta):** {'🟢 Listo' if s1_ok else '🟡 Pendiente (0 B)'}")
    st.caption(modules_status["ingest"]["details"])
    st.markdown(f"**Sprint 2 (Armonización):** {'🟢 Listo' if s2_ok else '🟡 Pendiente (0 B)'}")
    st.caption(modules_status["harmonize"]["details"])
    st.markdown(f"**Sprint 3 (Analítica):** {'🟢 Listo' if s3_ok else '🟡 Pendiente (0 B)'}")
    st.caption(modules_status["analytics"]["details"])
    st.markdown("**Sprint 4 (Visualizador Web):** 🟢 Operativo")

# 1. Selección Geográfica
st.sidebar.subheader("📍 1. Selección Geográfica")

geo_method = st.sidebar.radio(
    "Modalidad de selección:",
    options=["Región Predefinida", "Dibujar en Mapa", "Coordenadas Manuales"],
    index=0
)

if geo_method == "Región Predefinida":
    preset_choice = st.sidebar.selectbox(
        "Seleccione región geográfica:",
        options=list(REGIONES_PRESET.keys()),
        index=0
    )
    p_w, p_s, p_e, p_n = REGIONES_PRESET[preset_choice]
    st.session_state["lat_min"] = max(ds_lat_min, p_s)
    st.session_state["lat_max"] = min(ds_lat_max, p_n)
    st.session_state["lon_min"] = max(ds_lon_min, p_w)
    st.session_state["lon_max"] = min(ds_lon_max, p_e)

elif geo_method == "Dibujar en Mapa":
    st.sidebar.caption("Trace un rectángulo en el visor para capturar las coordenadas:")
    center_lat = (st.session_state["lat_min"] + st.session_state["lat_max"]) / 2
    center_lon = (st.session_state["lon_min"] + st.session_state["lon_max"]) / 2

    m_draw = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles="OpenStreetMap"
    )
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
    ).add_to(m_draw)

    map_draw_state = st_folium(
        m_draw,
        height=260,
        returned_objects=["last_active_drawing"],
        key="folium_draw_selector"
    )

    if map_draw_state and map_draw_state.get("last_active_drawing"):
        try:
            geom = shape(map_draw_state["last_active_drawing"]["geometry"])
            b = geom.bounds  # (minx, miny, maxx, maxy) -> (lon_min, lat_min, lon_max, lat_max)
            st.session_state["lon_min"] = float(round(b[0], 2))
            st.session_state["lat_min"] = float(round(b[1], 2))
            st.session_state["lon_max"] = float(round(b[2], 2))
            st.session_state["lat_max"] = float(round(b[3], 2))
            st.sidebar.success("Área dibujada capturada exitosamente.")
        except Exception:
            pass

elif geo_method == "Coordenadas Manuales":
    st.sidebar.caption("Defina los límites exactos en grados decimales (WGS84):")
    c1, c2 = st.sidebar.columns(2)
    st.session_state["lat_min"] = c1.number_input(
        "Lat Mín (Sur)",
        value=float(round(st.session_state["lat_min"], 2)),
        min_value=-90.0, max_value=90.0, step=0.25, format="%.2f"
    )
    st.session_state["lat_max"] = c2.number_input(
        "Lat Máx (Norte)",
        value=float(round(st.session_state["lat_max"], 2)),
        min_value=-90.0, max_value=90.0, step=0.25, format="%.2f"
    )

    c3, c4 = st.sidebar.columns(2)
    st.session_state["lon_min"] = c3.number_input(
        "Lon Mín (Oeste)",
        value=float(round(st.session_state["lon_min"], 2)),
        min_value=-180.0, max_value=180.0, step=0.25, format="%.2f"
    )
    st.session_state["lon_max"] = c4.number_input(
        "Lon Máx (Este)",
        value=float(round(st.session_state["lon_max"], 2)),
        min_value=-180.0, max_value=180.0, step=0.25, format="%.2f"
    )

active_lat_min = st.session_state["lat_min"]
active_lat_max = st.session_state["lat_max"]
active_lon_min = st.session_state["lon_min"]
active_lon_max = st.session_state["lon_max"]

# Validación de coordenadas
is_geo_valid, geo_err_msg = validate_coordinates(
    active_lat_min, active_lat_max, active_lon_min, active_lon_max
)
if not is_geo_valid:
    st.sidebar.error(f"❌ {geo_err_msg}")
    st.error(f"**Error en coordenadas geográficas:** {geo_err_msg}")
    st.stop()

active_bbox = (active_lon_min, active_lat_min, active_lon_max, active_lat_max)

# 2. Selección Temporal
st.sidebar.subheader("📅 2. Ventana Temporal")

c_date1, c_date2 = st.sidebar.columns(2)
date_start = c_date1.date_input("Desde:", value=ds_min_date, min_value=ds_min_date, max_value=ds_max_date)
date_end = c_date2.date_input("Hasta:", value=ds_max_date, min_value=ds_min_date, max_value=ds_max_date)

# Validación de fechas
is_date_valid, date_err_msg = validate_dates(date_start, date_end, ds_min_date, ds_max_date)
if not is_date_valid:
    st.sidebar.error(f"❌ {date_err_msg}")
    st.error(f"**Error en rango de fechas:** {date_err_msg}")
    st.stop()

# 3. Filtros opcionales por sensor y confianza
st.sidebar.subheader("🛰️ 3. Filtros de Detección (Opcional)")
selected_sources = st.sidebar.multiselect(
    "Sensores satelitales:",
    options=sorted(df_raw["source"].unique()),
    default=sorted(df_raw["source"].unique())
)
selected_confidences = st.sidebar.multiselect(
    "Nivel de confianza:",
    options=sorted(df_raw["confidence_level"].unique()),
    default=sorted(df_raw["confidence_level"].unique())
)

# Botón de confirmación para procesar
btn_confirmar = st.sidebar.button("⚡ Procesar / Actualizar Área", type="primary", use_container_width=True)

# ==========================================
# Orquestación del Pipeline (Sprints 1 → 2 → 3 → 4)
# ==========================================
pipeline_result = request_pipeline_execution(active_bbox, date_start, date_end)

# ==========================================
# Encabezado Principal y Avisos Metodológicos
# ==========================================
st.title("🔥 Calendario Armonizado de Actividad de Incendios")
st.markdown(
    "**NASA Space Apps Challenge 2026** — *Harmonization of MODIS and VIIRS Hot Spots*  \n"
    "Sistema integrado: Ingesta (Sprint 1) → Armonización (Sprint 2) → Analítica (Sprint 3) → Visualización (Sprint 4)."
)

# Identificación clara del dataset
st.caption(
    "📌 **Dataset Activo:** Serie histórica cerrada 2015–2020 | Cobertura: Colombia | "
    "Sensores: MODIS (Terra/Aqua 1 km) y VIIRS (Suomi-NPP 375 m) | NASA FIRMS"
)

# Aviso metodológico NASA
st.info(
    "🛰️ **Aviso científico NASA FIRMS:** Las cifras corresponden a **detecciones térmicas satelitales "
    "(focos de calor / hot spots)** armonizadas a nivel de píxel y deduplicadas espaciotemporalmente (1 km, 3 h). "
    "No deben interpretarse directamente como incendios individuales en tierra.",
    icon="ℹ️"
)

# Verificación de cobertura geográfica
if not pipeline_result["can_visualize"]:
    # El usuario seleccionó una región fuera de Colombia que requiere ejecutar los Sprints 1, 2 y 3
    st.warning("⚠️ **Atención: Área fuera de la cobertura precalculada de Colombia**")
    st.markdown(
        f"Ha seleccionado una región geográfica externa: `Lat: [{active_lat_min:.2f}°, {active_lat_max:.2f}°]`, "
        f"`Lon: [{active_lon_min:.2f}°, {active_lon_max:.2f}°]`."
    )
    st.error(
        "**Requisito de la profesora:** Para procesar una nueva área fuera del dataset nacional, "
        "el sistema debe ejecutar secuencialmente:  \n"
        "1. Ingesta (`ingest.py`) desde NASA FIRMS.  \n"
        "2. Armonización (`harmonize.py`) MODIS ↔ VIIRS.  \n"
        "3. Analítica (`analytics.py`) para agregaciones y anomalías.  \n\n"
        "**Estado actual:** Los scripts de los compañeros se encuentran actualmente **pendientes o con archivos vacíos (0 bytes)**."
    )

    with st.expander("📋 Ver contratos técnicos pendientes que deben implementar los otros integrantes", expanded=True):
        contracts = get_required_contracts_doc()
        for s_key, c_info in contracts.items():
            st.markdown(f"### {c_info['responsable']} (`{c_info['modulo']}`)")
            st.code(c_info["firma_requerida"], language="python")
            st.markdown(f"- **Salidas esperadas:** `{', '.join(c_info['salidas_esperadas'])}`")
            st.markdown(f"- **Regla científica:** {c_info['regla_cientifica']}")
            st.divider()

    st.info("💡 **Solución:** Utilice una región dentro de la cobertura de Colombia (2015–2020) para explorar el visor funcional.")
    if st.button("⬅️ Restablecer a Toda Colombia"):
        st.session_state["lat_min"] = ds_lat_min
        st.session_state["lat_max"] = ds_lat_max
        st.session_state["lon_min"] = ds_lon_min
        st.session_state["lon_max"] = ds_lon_max
        st.rerun()
    st.stop()

# ==========================================
# Filtrado de Datos en Memoria (Cobertura Activa)
# ==========================================
df_region = filter_bbox(df_raw, active_bbox)

if selected_sources:
    df_region = df_region[df_region["source"].isin(selected_sources)]
if selected_confidences:
    df_region = df_region[df_region["confidence_level"].isin(selected_confidences)]

df_filtered = filter_dates(df_region, date_start, date_end)

# Indicador de conteo en la barra lateral
st.sidebar.divider()
pct_total = (len(df_filtered) / len(df_raw) * 100) if len(df_raw) > 0 else 0.0
st.sidebar.metric(
    "Detecciones en consulta",
    f"{len(df_filtered):,}",
    delta=f"{pct_total:.1f}% del total nacional"
)

# Badge de consulta activa
is_full_coverage = (
    abs(active_lat_min - ds_lat_min) < 0.1 and abs(active_lat_max - ds_lat_max) < 0.1 and
    abs(active_lon_min - ds_lon_min) < 0.1 and abs(active_lon_max - ds_lon_max) < 0.1
)

c_b1, c_b2 = st.columns([2, 3])
with c_b1:
    st.markdown(
        f"**Área activa:** `Lat: [{active_lat_min:.2f}°, {active_lat_max:.2f}°] | Lon: [{active_lon_min:.2f}°, {active_lon_max:.2f}°]`  \n"
        f"*(Modo: {'Toda Colombia' if is_full_coverage else geo_method})*"
    )
with c_b2:
    days_count = (date_end - date_start).days + 1
    st.markdown(
        f"**Periodo activo:** `{date_start}` al `{date_end}` (`{days_count:,}` días)  \n"
        f"*(Línea base histórica calculada sobre el mismo ámbito geográfico)*"
    )

st.divider()

# Caso sin resultados
if len(df_filtered) == 0:
    st.warning(
        "⚠️ **Sin detecciones:** No se encontraron registros térmicos en las coordenadas y periodo seleccionados.  \n"
        "Sugerencia: Amplíe el área geográfica en la barra lateral o extienda el rango de fechas."
    )
    st.stop()

# ==========================================
# Cómputo Analítico (Coherencia Sprint 3)
# ==========================================
baseline_years = [y for y in [2015, 2016, 2017, 2018, 2019] if y in df_region["year"].unique()]
med_df_regional = historical_median(df_region, baseline_years=baseline_years)
anomalies_df = compute_anomalies(df_filtered, median_df=med_df_regional)
cal_matrix = calendar_matrix(df_filtered, all_years=sorted(df_filtered["year"].unique()))

# ==========================================
# Indicadores Clave (KPI Cards)
# ==========================================
k1, k2, k3, k4 = st.columns(4)

total_focos = len(df_filtered)
k1.metric("Total Detecciones", f"{total_focos:,}")

annual_sum = df_filtered.groupby("year").size()
if not annual_sum.empty:
    peak_year = annual_sum.idxmax()
    k2.metric("Año de Mayor Actividad", f"{peak_year}", f"{annual_sum.max():,} focos")
else:
    k2.metric("Año de Mayor Actividad", "N/A")

monthly_sum = monthly_counts(df_filtered).groupby("month_name")["count"].sum()
if not monthly_sum.empty:
    peak_month = monthly_sum.idxmax()
    k3.metric("Mes de Mayor Actividad", f"{peak_month}", f"{monthly_sum.max():,} focos")
else:
    k3.metric("Mes de Mayor Actividad", "N/A")

valid_anom = anomalies_df.dropna(subset=["anomaly_ratio"])
if not valid_anom.empty:
    last_row = valid_anom.iloc[-1]
    last_label = last_row["anomaly_label"]
    last_ratio_str = last_row["display_ratio"]
    k4.metric(
        f"Última Anomalía ({last_row['year_month']})",
        last_ratio_str,
        delta=f"Estado: {last_label}",
        delta_color="inverse" if "alta" in last_label else "normal"
    )
else:
    k4.metric("Última Anomalía", "Sin ref. histórica")

# ==========================================
# Pestañas de Visualización Principal
# ==========================================
tab_cal, tab_anom, tab_geo, tab_export, tab_pipeline = st.tabs([
    "📊 Calendario de Actividad (Heatmap)",
    "📈 Anomalías vs. Mediana Histórica",
    "🗺️ Mapa e Inspección Espacial",
    "💾 Exportación Segura de Datos",
    "🔗 Flujo de Integración (Sprints 1–4)"
])

# ------------------------------------------
# Pestaña 1: Calendario Histórico Heatmap
# ------------------------------------------
with tab_cal:
    st.subheader("📊 Calendario Anual de Actividad de Fuego")
    st.markdown(
        "Frecuencia mensual de detecciones térmicas organizadas en cuadrícula cronológica (año × 12 meses). "
        "Las celdas indican el número exacto de focos armonizados en la región seleccionada."
    )

    if not cal_matrix.empty:
        fig_heat = go.Figure(
            data=go.Heatmap(
                z=cal_matrix.values,
                x=cal_matrix.columns.tolist(),
                y=[f"Año {y}" for y in cal_matrix.index.tolist()],
                colorscale=HEATMAP_COLORSCALE,
                hovertemplate="<b>%{y} — Mes %{x}</b><br>Detecciones: %{z:,.0f} focos<extra></extra>",
                colorbar=dict(
                    title="Focos",
                    thickness=16,
                    len=0.9
                )
            )
        )
        fig_heat.update_layout(
            height=max(320, len(cal_matrix) * 45),
            margin=dict(l=40, r=40, t=50, b=40),
            xaxis=dict(title="Mes del Año", side="top", tickfont=dict(size=12)),
            yaxis=dict(title="Año", autorange="reversed", tickfont=dict(size=12)),
            font=dict(family="Arial, sans-serif")
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.caption(
            f"Total acumulado en el calendario: {int(cal_matrix.values.sum()):,} focos armonizados. "
            f"Año más activo en la selección: {cal_matrix.sum(axis=1).idxmax()} ({int(cal_matrix.sum(axis=1).max()):,} focos)."
        )
    else:
        st.info("Sin datos para generar la matriz de calendario.")

# ------------------------------------------
# Pestaña 2: Anomalías vs. Mediana Histórica
# ------------------------------------------
with tab_anom:
    st.subheader("📈 Anomalías Mensuales frente a la Mediana Histórica Regional")
    st.markdown(
        "Comparación de las detecciones de cada período respecto a la mediana histórica del mismo mes en la región. "
        "Permite identificar períodos excepcionalmente secos o activos."
    )

    col_u1, col_u2, col_u3 = st.columns(3)
    col_u1.markdown(f"🔴 **Alta:** Ratio ≥ `{ANOMALY_THRESHOLD_HIGH}x` (supera en 50% la mediana histórica)")
    col_u2.markdown(f"🟢 **Normal:** Ratio entre `{ANOMALY_THRESHOLD_LOW}x` y `{ANOMALY_THRESHOLD_HIGH}x`")
    col_u3.markdown(f"🔵 **Baja:** Ratio ≤ `{ANOMALY_THRESHOLD_LOW}x` (inferior a la mitad de la mediana)")

    if not anomalies_df.empty and not anomalies_df["anomaly_ratio"].dropna().empty:
        fig_bar = px.bar(
            anomalies_df,
            x="year_month",
            y="anomaly_ratio",
            color="anomaly_label",
            color_discrete_map=ANOMALY_COLOR_MAP,
            labels={
                "year_month": "Período (Año-Mes)",
                "anomaly_ratio": "Ratio vs. Mediana Histórica",
                "anomaly_label": "Clasificación"
            },
            hover_data={
                "count": ":,.0f",
                "median_historical": ":,.1f",
                "display_ratio": True,
                "anomaly_ratio": False
            }
        )

        fig_bar.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="#424242",
            line_width=2,
            annotation_text="Mediana Histórica (1.0x)",
            annotation_position="top left"
        )

        fig_bar.update_layout(
            height=390,
            margin=dict(l=40, r=40, t=30, b=60),
            xaxis_tickangle=-45,
            xaxis_title="Año - Mes",
            yaxis_title="Ratio de Anomalía",
            legend_title_text="Anomalía",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        with st.expander("🔍 Ver tabla de datos de anomalías y medianas de referencia"):
            st.dataframe(
                anomalies_df[[
                    "year_month", "count", "median_historical", "n_years", "display_ratio", "anomaly_label"
                ]].rename(columns={
                    "year_month": "Periodo",
                    "count": "Detecciones",
                    "median_historical": "Mediana Histórica",
                    "n_years": "Años Referencia",
                    "display_ratio": "Ratio",
                    "anomaly_label": "Categoría"
                }),
                use_container_width=True,
                hide_index=True
            )
    else:
        st.info("No hay suficientes registros históricos para calcular anomalías en la selección activa.")

# ------------------------------------------
# Pestaña 3: Inspección Geoespacial
# ------------------------------------------
with tab_geo:
    st.subheader("🗺️ Área de Consulta e Inspección Espacial")
    st.markdown(
        f"Límites geográficos activos: `[{active_lat_min:.2f}°, {active_lat_max:.2f}° N]`, "
        f"`[{active_lon_min:.2f}°, {active_lon_max:.2f}° O]`."
    )

    center_lat = (active_lat_min + active_lat_max) / 2
    center_lon = (active_lon_min + active_lon_max) / 2
    m_main = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="OpenStreetMap")

    folium.Rectangle(
        bounds=[[active_lat_min, active_lon_min], [active_lat_max, active_lon_max]],
        color="#D32F2F",
        weight=3,
        fill=True,
        fill_color="#FF8A80",
        fill_opacity=0.15,
        popup=f"Área activa: {len(df_filtered):,} focos",
        tooltip="Bounding Box de consulta activo"
    ).add_to(m_main)

    MAX_MAP_POINTS = 500
    if len(df_filtered) > MAX_MAP_POINTS:
        sample_df = df_filtered.sample(n=MAX_MAP_POINTS, random_state=42)
        st.caption(f"Mostrando una muestra representativa de {MAX_MAP_POINTS} detecciones de las {len(df_filtered):,} totales.")
    else:
        sample_df = df_filtered

    for _, pt in sample_df.iterrows():
        color_pt = "#D32F2F" if str(pt.get("source")) == "viirs-snpp" else "#FF9800"
        folium.CircleMarker(
            location=[float(pt["latitude"]), float(pt["longitude"])],
            radius=3,
            color=color_pt,
            fill=True,
            fill_color=color_pt,
            fill_opacity=0.7,
            popup=(
                f"<b>{pt['source']}</b><br>"
                f"Fecha: {pt['timestamp']}<br>"
                f"Confianza: {pt.get('confidence_level', 'N/A')}<br>"
                f"FRP: {pt.get('frp', 0.0)} MW"
            )
        ).add_to(m_main)

    st_folium(m_main, height=480, use_container_width=True, key="folium_main_preview")

    c_s1, c_s2 = st.columns(2)
    with c_s1:
        st.markdown("**Distribución por Sensor Satelital:**")
        st.dataframe(df_filtered["source"].value_counts().reset_index().rename(
            columns={"source": "Sensor", "count": "Detecciones"}
        ), use_container_width=True, hide_index=True)
    with c_s2:
        st.markdown("**Distribución por Nivel de Confianza Armonizado:**")
        st.dataframe(df_filtered["confidence_level"].value_counts().reset_index().rename(
            columns={"confidence_level": "Nivel Confianza", "count": "Detecciones"}
        ), use_container_width=True, hide_index=True)

# ------------------------------------------
# Pestaña 4: Exportación Segura de Datos
# ------------------------------------------
with tab_export:
    st.subheader("💾 Exportación de Resultados Filtrados")
    st.markdown(
        "Descargue el conjunto exacto de observaciones que coinciden con los filtros geográficos y temporales activos.  \n"
        "**Protección de seguridad:** Se aplica sanitización contra *CSV Injection* para prevenir la ejecución de fórmulas "
        "al abrir el archivo en hojas de cálculo (Excel, LibreOffice)."
    )

    export_cols = [
        "latitude", "longitude", "acq_date", "acq_time", "timestamp",
        "satellite", "instrument", "source", "confidence_level", "frp", "daynight"
    ]
    available_export_cols = [c for c in export_cols if c in df_filtered.columns]

    df_export_safe = sanitize_for_csv(df_filtered, columns=available_export_cols)

    st.markdown(f"**Vista previa de los primeros 10 registros (de {len(df_export_safe):,} totales):**")
    st.dataframe(df_export_safe.head(10), use_container_width=True, hide_index=True)

    csv_bytes = df_export_safe.to_csv(index=False).encode("utf-8")
    filename = (
        f"focos_armonizados_lat_{active_lat_min:.2f}_{active_lat_max:.2f}_"
        f"lon_{active_lon_min:.2f}_{active_lon_max:.2f}_{date_start}_{date_end}.csv"
    )

    st.download_button(
        label=f"📥 Descargar CSV ({len(df_export_safe):,} registros, ~{len(csv_bytes) / 1024:.1f} KB)",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        type="primary"
    )

# ------------------------------------------
# Pestaña 5: Flujo de Integración (Sprints 1–4)
# ------------------------------------------
with tab_pipeline:
    st.subheader("🔗 Arquitectura de Integración de Sprints")
    st.markdown(
        "Conexión del flujo de datos solicitada por la profesora:  \n"
        "`Selección de Región (Sprint 4) → Ingesta (Sprint 1) → Armonización (Sprint 2) → Analítica (Sprint 3) → Visor (Sprint 4)`"
    )

    st.markdown("#### Estado Actual de los Módulos en el Repositorio:")
    status_table = []
    for mod_key, info in modules_status.items():
        status_table.append({
            "Módulo": info["expected_file"],
            "Ruta": str(info["path"].name),
            "Tamaño": f"{info['size_bytes']} bytes",
            "Estado": "🟢 Operativo" if info["has_required_interface"] else ("🟡 Archivo Vacío (0 B)" if info["size_bytes"] == 0 else "🔴 Sin interfaz parametrizada"),
            "Diagnóstico": info["details"]
        })
    st.dataframe(pd.DataFrame(status_table), use_container_width=True, hide_index=True)

    st.markdown("#### Contratos Técnicos de Integración:")
    st.caption("Especificación que deben satisfacer los scripts de los compañeros para conectarse a la llamada dinámica:")
    contracts = get_required_contracts_doc()
    for s_key, c_info in contracts.items():
        with st.expander(f"📌 {c_info['responsable']} — {c_info['modulo']}"):
            st.markdown(f"**Firma de función requerida:**")
            st.code(c_info["firma_requerida"], language="python")
            st.markdown(f"**Parámetros:**")
            for param, desc in c_info["parametros"].items():
                st.markdown(f"- `{param}`: {desc}")
            st.markdown(f"**Salidas esperadas:**")
            for out in c_info["salidas_esperadas"]:
                st.markdown(f"- `{out}`")
            st.markdown(f"**Regla metodológica:** {c_info['regla_cientifica']}")

# ==========================================
# Pie de página y trazabilidad
# ==========================================
st.divider()
st.caption(
    "🔥 NASA FIRMS Harmonized Fire Activity Calendar | Sprint 4 MVP | "
    "Sensores: Terra/Aqua (MODIS 1km) y Suomi-NPP (VIIRS 375m) | "
    "Flujo integrado con adaptador de pipeline para Sprints 1–3."
)
