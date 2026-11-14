# 🔥 Calendario Armonizado de Actividad de Incendios

Armonización de MODIS y VIIRS (NASA FIRMS) para visualizar patrones históricos de incendios.

## El problema
MODIS (1 km, confianza 0-100) y VIIRS (375 m, confianza categórica) no son comparables directamente. Sin una serie armonizada, los gestores de emergencias no pueden ver patrones históricos coherentes.

## La solución
1. Filtrado de solo incendios de vegetación (`type == 0`)
2. Normalización de confianza a escala `low`/`nominal`/`high`
3. Deduplicación espaciotemporal (1 km, 3 h)
4. App web con calendario, anomalías y descarga CSV

## Stack
- Ingesta: `firms-client` (sin API key)
- Procesamiento: pandas, numpy
- App: Streamlit + Plotly + Folium
- Validación: contraste con IDEAM

## Reproducibilidad
```bash
pip install -r requirements.txt
cd src
python ingest.py && python harmonize.py && python analytics.py
streamlit run app.py
```
