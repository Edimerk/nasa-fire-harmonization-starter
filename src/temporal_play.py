"""Reproducción temporal global de las detecciones ya consultadas en FireTrackers.

El origen puede ser la base armonizada de Colombia o cualquier resultado de
NASA FIRMS procesado mediante el pipeline de Sprints 1 -> 2 -> 3.
Nunca realiza solicitudes a la API por cada cuadro; los periodos representan
solo la cobertura de fechas que la consulta seleccionada pudo obtener.
"""
from __future__ import annotations

import datetime as dt
import math
from collections import OrderedDict

import pandas as pd
import plotly.graph_objects as go

MAX_FRAMES = 84
MAX_POINTS_PER_FRAME = 175
MAX_ACCUMULATED_POINTS = 900
FREQUENCIES = {"Horaria": "h", "Diaria": "D", "Semanal": "W-SUN", "Mensual": "M", "Anual": "Y"}


def default_frequency(start: dt.date, end: dt.date) -> str:
    days = (end - start).days + 1
    if days <= 3:
        return "Horaria"
    if days <= MAX_FRAMES:
        return "Diaria"
    if days <= MAX_FRAMES * 7:
        return "Semanal"
    if days <= 365 * 6:
        return "Mensual"
    return "Anual"


def period_labels(start: dt.date, end: dt.date, frequency: str) -> list[str]:
    if start > end:
        raise ValueError("La fecha inicial es posterior a la final.")
    if frequency not in FREQUENCIES:
        raise ValueError("Frecuencia temporal no válida.")
    period_end = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    labels = [str(v) for v in pd.period_range(start=start, end=period_end, freq=FREQUENCIES[frequency])]
    if len(labels) > MAX_FRAMES:
        raise ValueError(f"Se necesitan {len(labels)} cuadros, por encima del límite de {MAX_FRAMES}. "
                         "Selecciona una agrupación más amplia (semanal, mensual o anual).")
    return labels


def _prepare_data(df: pd.DataFrame, bbox, start: dt.date, end: dt.date, frequency: str) -> pd.DataFrame:
    missing = [c for c in ("timestamp", "latitude", "longitude") if c not in df.columns]
    if missing:
        raise ValueError("Faltan columnas necesarias: " + ", ".join(missing))
    cols = [c for c in ("timestamp", "latitude", "longitude", "source", "frp") if c in df.columns]
    data = df[cols].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    data["latitude"] = pd.to_numeric(data["latitude"], errors="coerce")
    data["longitude"] = pd.to_numeric(data["longitude"], errors="coerce")
    data = data.dropna(subset=["timestamp", "latitude", "longitude"])
    west, south, east, north = bbox
    data = data.loc[
        data["longitude"].between(west, east) & data["latitude"].between(south, north) &
        (data["timestamp"] >= pd.Timestamp(start, tz="UTC")) &
        (data["timestamp"] < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1))
    ].copy()
    if "source" not in data.columns:
        data["source"] = "Sin especificar"
    data["source"] = data["source"].astype(str).replace({"nan": "Sin especificar"})
    if "frp" not in data.columns:
        data["frp"] = float("nan")
    data["frp"] = pd.to_numeric(data["frp"], errors="coerce")
    data["_frame"] = data["timestamp"].dt.tz_convert(None).dt.to_period(FREQUENCIES[frequency]).astype(str)
    return data


def _trace(data: pd.DataFrame) -> go.Scattermap:
    if data.empty:
        return go.Scattermap(lat=[], lon=[], mode="markers", hoverinfo="skip", showlegend=False)
    hover = []
    color = []
    for source, when, frp in zip(data["source"], data["timestamp"], data["frp"]):
        source_name = str(source)
        color.append("#F97316" if "modis" in source_name.lower() else "#DC2626")
        frp_text = "N/D" if pd.isna(frp) else f"{frp:.1f} MW"
        hover.append(f"{source_name} · {when.strftime('%Y-%m-%d %H:%M UTC')}<br>FRP: {frp_text}")
    return go.Scattermap(
        lat=data["latitude"].tolist(), lon=data["longitude"].tolist(),
        mode="markers", marker=dict(size=8, color=color, opacity=.8),
        hovertext=hover, hoverinfo="text", showlegend=False,
    )


def build_timeline_figure(
    df: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    start: dt.date,
    end: dt.date,
    *,
    frequency: str = "Automática",
    mode: str = "Por periodo",
    speed_ms: int = 550,
) -> tuple[go.Figure, dict]:
    """Representa cualquier BBOX y rango YA cargado, sin nuevas consultas FIRMS."""
    if len(bbox) != 4 or not (-180 <= bbox[0] < bbox[2] <= 180 and -90 <= bbox[1] < bbox[3] <= 90):
        raise ValueError("El BBOX de la consulta no es válido.")
    if mode not in ("Por periodo", "Acumulado"):
        raise ValueError("Modo de reproducción no válido.")
    if frequency == "Automática":
        frequency = default_frequency(start, end)
    labels = period_labels(start, end, frequency)
    data = _prepare_data(df, bbox, start, end, frequency)
    counts = data["_frame"].value_counts().to_dict()
    real_counts = OrderedDict((label, int(counts.get(label, 0))) for label in labels)
    display_counts = OrderedDict()
    cumsum = 0
    for label, count in real_counts.items():
        cumsum += count
        display_counts[label] = cumsum if mode == "Acumulado" else count

    # El muestreo se aplica solo a los puntos de mapa, nunca a los conteos.
    if mode == "Acumulado":
        sampled = data.sample(min(len(data), MAX_ACCUMULATED_POINTS), random_state=53) if len(data) else data
    else:
        # Muestreo una vez por periodo, sin repetir grandes filtros sobre la base original.
        groups = []
        for _, group in data.groupby("_frame", sort=False):
            if len(group) > MAX_POINTS_PER_FRAME:
                group = group.sample(n=MAX_POINTS_PER_FRAME, random_state=53)
            groups.append(group)
        sampled = pd.concat(groups, ignore_index=True) if groups else data

    samples_by_period = {key: group for key, group in sampled.groupby("_frame", sort=False)}
    blank = sampled.iloc[:0]
    frame_traces = []
    cumulative_parts = []
    for label in labels:
        current = samples_by_period.get(label, blank)
        if mode == "Acumulado":
            if not current.empty:
                cumulative_parts.append(current)
            current = pd.concat(cumulative_parts, ignore_index=True) if cumulative_parts else blank
        frame_traces.append(_trace(current))

    frames = [go.Frame(
        name=label, data=[trace],
        layout=go.Layout(title=dict(text=f"{label} · {display_counts[label]:,} detecciones"))
    ) for label, trace in zip(labels, frame_traces)]

    west, south, east, north = bbox
    lon_span = max(east - west, 0.05)
    lat_span = max(north - south, 0.05)
    zoom = max(1, min(10, math.log2(360 / max(lon_span * 2.2, lat_span * 3.1))))
    fig = go.Figure(data=[frame_traces[0]], frames=frames)
    fig.update_layout(
        title=dict(text=f"{labels[0]} · {display_counts[labels[0]]:,} detecciones", font=dict(size=14)),
        map=dict(
            style="open-street-map",
            center=dict(lon=(west + east) / 2, lat=(south + north) / 2),
            zoom=zoom,
        ),
        height=480,
        margin=dict(l=0, r=0, t=42, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        updatemenus=[dict(
            type="buttons", showactive=False, direction="left",
            x=0.01, xanchor="left", y=0, yanchor="top", pad=dict(t=42),
            buttons=[
                dict(label="▶ Play", method="animate", args=[None, dict(
                    frame=dict(duration=int(speed_ms), redraw=True),
                    transition=dict(duration=0), fromcurrent=True, mode="immediate")]),
                dict(label="⏸ Pausa", method="animate", args=[[None], dict(
                    frame=dict(duration=0, redraw=False),
                    transition=dict(duration=0), mode="immediate")]),
            ],
        )],
        sliders=[dict(
            active=0, x=.18, len=.8, y=0, yanchor="top", pad=dict(b=0, t=42),
            currentvalue=dict(prefix="Fecha: ", visible=True),
            steps=[dict(label=label, method="animate", args=[[label], dict(
                mode="immediate", frame=dict(duration=0, redraw=True),
                transition=dict(duration=0))]) for label in labels],
        )],
    )
    info = {
        "frequency": frequency,
        "frames": len(labels),
        "total_records": len(data),
        "counts_by_period": real_counts,
        "display_counts": display_counts,
        "sample_limited": len(data) > (MAX_ACCUMULATED_POINTS if mode == "Acumulado" else MAX_POINTS_PER_FRAME),
    }
    return fig, info


def render_player(df: pd.DataFrame, bbox, start: dt.date, end: dt.date) -> None:
    import streamlit as st

    st.caption("▶ Reproduce la región seleccionada, con los datos ya descargados de NASA y procesados "
               "por los Sprints 1, 2 y 3. Funciona para cualquier país disponible en la consulta.")
    if df is None or df.empty:
        st.info("La consulta actual no contiene detecciones para reproducir.")
        return
    if not st.checkbox("Activar reproducción temporal", value=False, key="ft_play_global_activar"):
        st.info("Activa la reproducción para mostrar Play, pausa y la barra de tiempo.")
        return
    options = ["Automática", "Horaria", "Diaria", "Semanal", "Mensual", "Anual"]
    col1, col2 = st.columns(2)
    with col1:
        frequency = st.selectbox("Escala temporal", options, key="ft_play_global_freq")
    with col2:
        mode = st.selectbox("Tipo de recorrido", ["Por periodo", "Acumulado"], key="ft_play_global_mode")
    speed = st.select_slider("Velocidad", options=["Lenta", "Normal", "Rápida"],
                             value="Normal", key="ft_play_global_speed")
    speed_ms = {"Lenta": 1150, "Normal": 550, "Rápida": 260}[speed]
    try:
        with st.spinner("Preparando recorrido temporal..."):
            fig, info = build_timeline_figure(
                df, tuple(bbox), start, end,
                frequency=frequency, mode=mode, speed_ms=speed_ms)
    except (ValueError, TypeError) as exc:
        st.warning(str(exc))
        return
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "scrollZoom": True})
    st.caption(f"{info['frames']} cuadros ({info['frequency']}) · "
               f"{info['total_records']:,} detecciones en el rango consultado. "
               "Usa el icono de pantalla completa en la gráfica para ampliar el mapa. "
               "Los conteos incluyen todos los registros; los marcadores pueden ser una muestra. "
               "No representan incendios únicos. Los periodos sin detecciones no certifican ausencia de incendios.")
