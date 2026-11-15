"""FireTrackers: concentracion espacial OBSERVADA (no riesgo predictivo).

Usa exclusivamente los datos armonizados del BBOX y las fechas seleccionadas.
No consulta NASA, no modifica ni recalcula los sprints existentes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_activity_zones(df: pd.DataFrame, bbox, grid_size: int = 5) -> pd.DataFrame:
    """Produce una cuadricula completa de conteos y densidad por km2 aproximado."""
    if grid_size < 2 or grid_size > 20:
        raise ValueError("grid_size debe estar entre 2 y 20")
    west, south, east, north = (float(x) for x in bbox)
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("BBOX invalido")
    if df is None or not {"latitude", "longitude"}.issubset(df.columns):
        raise ValueError("Faltan columnas latitude o longitude")

    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    valid = lat.notna() & lon.notna() & lat.between(south, north) & lon.between(west, east)
    lat_v = lat.loc[valid].to_numpy(dtype=float)
    lon_v = lon.loc[valid].to_numpy(dtype=float)
    row = np.minimum(grid_size - 1, np.floor((lat_v - south) / (north - south) * grid_size).astype(int))
    col = np.minimum(grid_size - 1, np.floor((lon_v - west) / (east - west) * grid_size).astype(int))
    counts = np.bincount(row * grid_size + col, minlength=grid_size * grid_size)
    n = len(lat_v)
    delta_lat = (north - south) / grid_size
    delta_lon = (east - west) / grid_size
    rows = []
    for r in range(grid_size):
        lat_min = south + r * delta_lat
        lat_max = lat_min + delta_lat
        lat_mid = (lat_min + lat_max) / 2
        # Aproximacion geodesica para comparaciones locales; no apta para zonas polares.
        area = max(1e-6, (111.32 * delta_lat) * (111.32 * delta_lon * abs(np.cos(np.deg2rad(lat_mid)))))
        for c in range(grid_size):
            lon_min = west + c * delta_lon
            lon_max = lon_min + delta_lon
            count = int(counts[r * grid_size + c])
            rows.append({
                "fila": r, "columna": c, "detecciones": count,
                "porcentaje": round(100.0 * count / n, 2) if n else 0.0,
                "densidad_1000km2": round(1000.0 * count / area, 3),
                "lat_centro": round(lat_mid, 4),
                "lon_centro": round((lon_min + lon_max) / 2, 4),
                "lat_min": round(lat_min, 4), "lat_max": round(lat_max, 4),
                "lon_min": round(lon_min, 4), "lon_max": round(lon_max, 4),
            })
    return pd.DataFrame(rows)


def render_activity_zones(df: pd.DataFrame, bbox, grid_size: int = 5) -> None:
    """Presenta calor espacial, ranking y CSV sin alterar la interfaz principal."""
    import streamlit as st
    import plotly.graph_objects as go
    st.markdown("**Zonas con mayor actividad observada**")
    st.caption("Distribucion espacial de las detecciones armonizadas en la zona y fechas seleccionadas.")
    try:
        summary = summarize_activity_zones(df, bbox, grid_size=grid_size)
    except (ValueError, TypeError) as exc:
        st.info(f"No se puede calcular la distribucion espacial: {exc}")
        return
    total = int(summary["detecciones"].sum())
    if not total:
        st.info("No hay detecciones validas para comparar zonas en este intervalo.")
        return
    matrix = summary.pivot(index="fila", columns="columna", values="detecciones").iloc[::-1]
    fig = go.Figure(go.Heatmap(
        z=matrix.to_numpy(), x=[f"Oeste→Este {i+1}" for i in matrix.columns],
        y=[f"Norte→Sur {i+1}" for i in range(grid_size)],
        colorscale="YlOrRd", colorbar={"title": "Registros", "thickness": 10},
        hovertemplate="Celda: %{x}, %{y}<br>Detecciones: %{z:,.0f}<extra></extra>",
    ))
    fig.update_layout(height=310, margin={"l": 10, "r": 5, "t": 10, "b": 30},
                      xaxis={"tickangle": -35, "tickfont": {"size": 9}},
                      yaxis={"tickfont": {"size": 10}})
    st.plotly_chart(fig, width="stretch")
    top = (summary.loc[summary["detecciones"] > 0]
           .sort_values(["densidad_1000km2", "detecciones"], ascending=False)
           .head(5).copy())
    top.insert(0, "prioridad_relativa", [f"#{i}" for i in range(1, len(top) + 1)])
    st.caption("**Cinco celdas con mayor concentracion por superficie** (dentro de esta consulta).")
    st.dataframe(top[["prioridad_relativa", "detecciones", "porcentaje", "densidad_1000km2", "lat_centro", "lon_centro"]].rename(columns={
        "prioridad_relativa": "Posicion", "detecciones": "Detecciones",
        "porcentaje": "% del total", "densidad_1000km2": "Det./1000 km²",
        "lat_centro": "Lat. centro", "lon_centro": "Lon. centro",
    }), hide_index=True, width="stretch")
    st.download_button("📥 Descargar zonas (CSV)", data=summary.to_csv(index=False).encode("utf-8"),
                       file_name="firetrackers_zonas_actividad.csv", mime="text/csv", key="download_activity_zones")
    st.caption("**No es un mapa de riesgo ni un pronostico.** Compara detecciones observadas, no incendios confirmados, probabilidad de incendio o superficie quemada. La densidad por area es aproximada y depende de la cobertura y de los sensores.")
