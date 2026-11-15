"""Variacion RELATIVA interna a un periodo, distinta de una anomalia historica.

No consulta la API ni modifica las detecciones armonizadas. Cuenta dias con cero
solo si esos dias forman parte de un intervalo efectivamente consultado.
"""
from __future__ import annotations

import pandas as pd


def calcular_variaciones_periodo(df, inicio, fin, agrupacion="Semanal"):
    """Retorna (tabla, mediana) de tasas diarias de detecciones por bloque.

    La mediana se obtiene de los bloques dentro del mismo intervalo: NO es una
    referencia historica independiente, no indica riesgo ni pronostico.
    """
    inicio = pd.Timestamp(inicio).normalize()
    fin = pd.Timestamp(fin).normalize()
    if inicio > fin:
        raise ValueError("Fecha inicial posterior a la final")
    dias = (fin - inicio).days + 1
    if dias > 3660:
        raise ValueError("Intervalo demasiado largo")
    if agrupacion not in ("Diaria", "Semanal", "Mensual"):
        raise ValueError("Agrupacion desconocida")
    if "timestamp" not in df.columns:
        raise ValueError("Falta la columna timestamp")

    # Normalize UTC or naïve timestamps consistently (calendar date UTC).
    valores = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    fechas = valores.dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    fechas = fechas[(fechas >= inicio) & (fechas <= fin)]
    dias_indice = pd.date_range(inicio, fin, freq="D")
    conteos = fechas.value_counts().reindex(dias_indice, fill_value=0).sort_index()
    diaria = pd.DataFrame({"fecha": dias_indice, "conteo": conteos.to_numpy()})

    if agrupacion == "Diaria":
        diaria["grupo"] = diaria["fecha"]
    elif agrupacion == "Semanal":
        # Periodos consecutivos de siete dias contados desde el inicio.
        offset = (diaria["fecha"] - inicio).dt.days // 7
        diaria["grupo"] = inicio + pd.to_timedelta(offset * 7, unit="D")
    else:
        diaria["grupo"] = diaria["fecha"].dt.to_period("M").dt.to_timestamp()

    tabla = (diaria.groupby("grupo", as_index=False)
             .agg(detecciones=("conteo", "sum"), dias=("conteo", "size"))
             .rename(columns={"grupo": "periodo"}))
    tabla["tasa_diaria"] = tabla["detecciones"] / tabla["dias"]
    mediana = float(tabla["tasa_diaria"].median()) if not tabla.empty else 0.0
    tabla["diferencia_diaria"] = tabla["tasa_diaria"] - mediana
    if mediana > 0:
        tabla["variacion_pct"] = (100 * tabla["diferencia_diaria"] / mediana).round(1)
    else:
        tabla["variacion_pct"] = float("nan")
    tabla["categoria"] = "Cerca de la mediana del periodo"
    if mediana > 0:
        tabla.loc[tabla["variacion_pct"] >= 50, "categoria"] = "Mayor actividad relativa"
        tabla.loc[tabla["variacion_pct"] <= -50, "categoria"] = "Menor actividad relativa"
    else:
        tabla.loc[tabla["tasa_diaria"] > 0, "categoria"] = "Actividad con referencia cero"
    return tabla, mediana


def render_period_anomalies(df, inicio, fin):
    """UI de Streamlit invocada desde la pestana existente Anomalias."""
    import streamlit as st
    import plotly.express as px

    inicio_ts = pd.Timestamp(inicio).normalize()
    fin_ts = pd.Timestamp(fin).normalize()
    numero_dias = (fin_ts - inicio_ts).days + 1
    st.markdown("**Variaciones de actividad dentro del periodo consultado**")
    st.caption(
        "Cada dia o semana se compara con la mediana de ESTE intervalo. "
        "No es una anomalia historica ni una prediccion de incendios."
    )
    if numero_dias < 7:
        st.info("Selecciona al menos 7 dias para comparar periodos. No se calcula una referencia con pocos dias.")
        return
    if df.empty:
        st.info("No hay detecciones para comparar en el periodo seleccionado.")
        return
    opciones = ["Diaria", "Semanal"] + (["Mensual"] if numero_dias >= 45 else [])
    sugerida = 1 if numero_dias >= 22 else 0
    frecuencia = st.selectbox("Comparar por", opciones, index=sugerida,
                              key="firetrackers_period_anom_freq")
    tabla, mediana = calcular_variaciones_periodo(df, inicio, fin, frecuencia)
    if len(tabla) < 3:
        st.info("Esta agrupacion genera menos de 3 periodos comparables. Elige una escala mas fina.")
        return

    max_row = tabla.loc[tabla["tasa_diaria"].idxmax()]
    a, b = st.columns(2)
    a.metric("Mediana (detecciones/dia)", f"{mediana:,.1f}")
    b.metric("Periodo con mayor actividad", str(max_row["periodo"].date()))
    tabla_grafica = tabla.copy()
    tabla_grafica["Periodo"] = tabla_grafica["periodo"].dt.strftime("%Y-%m-%d")
    fig = px.bar(
        tabla_grafica,
        x="Periodo", y="tasa_diaria", color="categoria",
        color_discrete_map={
            "Mayor actividad relativa": "#E85D24",
            "Menor actividad relativa": "#4B82B5",
            "Cerca de la mediana del periodo": "#72A57B",
            "Actividad con referencia cero": "#E85D24",
        },
        hover_data={"detecciones": True, "dias": True, "variacion_pct": True,
                    "tasa_diaria": ":.2f"},
        labels={"tasa_diaria": "Detecciones/dia", "categoria": "Variacion"},
    )
    fig.add_hline(y=mediana, line_dash="dash", line_color="#334155",
                  annotation_text="Mediana del periodo")
    fig.update_layout(height=275, margin=dict(l=0, r=0, t=15, b=55),
                      showlegend=False, xaxis_tickangle=-45)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Metodo descriptivo: periodos del mismo BBOX e intervalo; semanas o meses "
        "incompletos se normalizan por dia. Los dias sin detecciones cuentan como "
        "cero solo cuando la descarga de esos dias fue completa. "
        "No comparar con climatologia multianual sin datos equivalentes."
    )
