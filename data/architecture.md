# Arquitectura y contratos preliminares

1. `ingest.py`: consume datos FIRMS y escribe CSV crudos y procesados preservando procedencia.
2. `harmonize.py`: ingiere datos procesados, conserva `source`, normaliza esquemas y aplica una metodología evaluada. La deduplicación **no basta** para equivalencia entre sensores.
3. `analytics.py`: calcula métricas por región, fecha, referencia histórica y anomalías con manejo de meses sin observaciones.
4. `app/main.py`: selecciona región y fechas, visualiza heatmap y permite exportar CSV.
5. `validate.py`: pruebas internas y, si hay datos compatibles, comparación externa con IDEAM.

## Reglas de integración
- El código importable no debe ejecutar descargas, abrir una app o guardar archivos como efecto secundario.
- Todas las rutas relativas a la raíz del repositorio; nada de `/content` fijo.
- Registrar fuente, nombre de producto, periodo, filtros, versión y fecha de descarga.
- Procesar datos de muestra antes de usar datasets nacionales masivos.
- No presentar detecciones como incendios individuales.
