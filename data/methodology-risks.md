# Riesgos y decisiones pendientes

1. **Armonización:** mapear confianzas y deduplicar 1 km / 3 h es una hipótesis, no una demostración de equivalencia de conteos históricos. Comparar en solapamiento y separar resultados por sensor hasta demostrar comparabilidad.
2. **Deduplicación:** el prototipo ordena por prioridad/FRP y compara solo los próximos 499 registros: no garantiza detección de vecinos. Diseñar búsqueda espacial/temporal indexada y pruebas con casos límite.
3. **Disponibilidad:** comprobar que el cliente FIRMS y columnas como `type` existen en ambos productos reales. No convertir valores desconocidos de confianza automáticamente a nominal.
4. **Cobertura:** los documentos y prototipo usan Colombia 2015–2020; no afirmar cobertura mundial 2000–presente sin datos y validación.
5. **Línea base:** excluir año actual incompleto, considerar meses ausentes, mediana cero y comparabilidad por observaciones disponibles.
6. **Validación externa:** contrastar procedencia IDEAM para evitar suponer independencia si reutiliza detecciones satelitales de FIRMS.
7. **Accesibilidad:** probar contraste sobre fondos reales y evitar transmitir anomalías solo con rojo/verde.
8. **Entregables:** comprobar formulario y normas NASA vigentes; documentos internos no sustituyen reglas oficiales.
9. **Datos grandes:** mantener datasets crudos fuera de Git; proporcionar muestras pequeñas redistribuibles y pasos de reproducción.
