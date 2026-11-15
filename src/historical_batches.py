"""Consulta histórica acotada con caché de bloques de cinco días.

No altera la armonización ni la analítica; devuelve CSV por sensor con el
mismo contrato de ``ingest_historico.ejecutar``.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import date, timedelta
from pathlib import Path

from ingest import BASE_DIR, COLUMNAS

MAX_DIAS_HISTORICOS = 93
BLOQUE_DIAS = 5


def dividir_intervalo(inicio: date, fin: date):
    """Rangos inclusivos, consecutivos, sin días repetidos ni omitidos."""
    if inicio > fin:
        raise ValueError("La fecha inicial debe ser anterior o igual a la final.")
    cursor = inicio
    while cursor <= fin:
        cierre = min(cursor + timedelta(days=BLOQUE_DIAS - 1), fin)
        yield cursor, cierre
        cursor = cierre + timedelta(days=1)


def _consulta_dir(bbox, inicio, fin):
    # El adaptador usa exactamente esta clave al ubicar la armonización.
    clave = f"{bbox[0]:.2f}_{bbox[1]:.2f}_{bbox[2]:.2f}_{bbox[3]:.2f}_{inicio}_{fin}"
    return BASE_DIR / "data" / "queries" / hashlib.md5(clave.encode("utf-8")).hexdigest()[:12]


def _resultado_valido(meta: dict, bbox, inicio, fin) -> bool:
    if meta.get("bbox") != list(bbox) or meta.get("inicio") != inicio.isoformat() or meta.get("fin") != fin.isoformat():
        return False
    rutas = meta.get("archivos", {})
    if not rutas:
        return False
    return all(Path(p).is_file() and Path(p).stat().st_size > 0 for p in rutas.values())


def descargar_historico_por_bloques(bbox, inicio: date, fin: date) -> dict:
    """Descarga rangos cortos, cachea bloques y une CSV sin acumular filas en RAM.

    Primera fase de seguridad: máximo 93 días y un BBOX <= 20° por dimensión.
    Para series multianuales se requiere un gestor de trabajos desacoplado de Streamlit.
    """
    bbox = tuple(float(v) for v in bbox)
    if len(bbox) != 4 or not (-180 <= bbox[0] < bbox[2] <= 180 and -90 <= bbox[1] < bbox[3] <= 90):
        raise ValueError("BBOX inválido.")
    if inicio > fin or fin > date.today():
        raise ValueError("Rango de fechas inválido.")
    dias = (fin - inicio).days + 1
    if dias > MAX_DIAS_HISTORICOS:
        raise ValueError(f"Esta primera fase permite hasta {MAX_DIAS_HISTORICOS} días por consulta externa. Selecciona un intervalo menor.")
    if dias > 5 and (bbox[2] - bbox[0] > 20 or bbox[3] - bbox[1] > 20):
        raise ValueError("Para más de cinco días, delimita una región de hasta 20° de ancho y 20° de alto.")

    carpeta = _consulta_dir(bbox, inicio, fin)
    manifest = carpeta / "consulta_bloques.json"
    if manifest.exists():
        try:
            previo = json.loads(manifest.read_text(encoding="utf-8"))
            if _resultado_valido(previo, bbox, inicio, fin):
                print(f"[CACHÉ HISTÓRICA] {inicio} / {fin}")
                return previo
        except (ValueError, OSError, TypeError):
            pass

    # Importación tardía: los tests pueden simular el módulo sin conectarse a NASA.
    from ingest_historico import ejecutar

    carpeta_procesados = carpeta / "processed"
    carpeta_procesados.mkdir(parents=True, exist_ok=True)
    # Nunca publicamos archivos agregados incompletos como válidos.
    temporales = {}
    nombres = {}
    resumen = {}
    estados = {}
    total_bloques = (dias + BLOQUE_DIAS - 1) // BLOQUE_DIAS
    procesados = 0

    try:
        for n, (desde, hasta) in enumerate(dividir_intervalo(inicio, fin), 1):
            # Cache permanente por bloque en ingest_historico: está dentro de data/queries.
            clave_bloque = hashlib.sha256(f"{bbox}|{desde}|{hasta}|historico".encode()).hexdigest()[:12]
            manifest_bloque = BASE_DIR / "data" / "queries" / clave_bloque / "consulta.json"
            bloque = None
            if manifest_bloque.exists():
                try:
                    candidato = json.loads(manifest_bloque.read_text(encoding="utf-8"))
                    if _resultado_valido(candidato, bbox, desde, hasta):
                        bloque = candidato
                except (ValueError, OSError, TypeError):
                    pass
            if bloque is None:
                bloque = ejecutar(bbox, desde, hasta)
            if not _resultado_valido(bloque, bbox, desde, hasta):
                raise RuntimeError(f"El bloque {desde} a {hasta} no generó CSV verificables.")

            for sensor, ruta in bloque["archivos"].items():
                if sensor not in temporales:
                    salida_tmp = carpeta_procesados / f".{sensor}_filtrado.tmp"
                    salida_tmp.write_text(",".join(COLUMNAS) + "\n", encoding="utf-8")
                    temporales[sensor] = salida_tmp
                    nombres[sensor] = str(carpeta_procesados / f"{sensor}_filtrado.csv")
                    resumen[sensor] = {"recibidos": 0, "limpios": 0, "descartados": 0, "sin_tipo": 0}
                    estados[sensor] = []
                with Path(ruta).open(encoding="utf-8-sig", newline="") as fh:
                    lector = csv.DictReader(fh)
                    if not lector.fieldnames or not set(COLUMNAS).issubset(set(lector.fieldnames)):
                        raise RuntimeError(f"CSV de {sensor} sin las columnas requeridas.")
                    with temporales[sensor].open("a", encoding="utf-8", newline="") as out:
                        writer = csv.DictWriter(out, fieldnames=COLUMNAS, extrasaction="ignore")
                        for fila in lector:
                            writer.writerow(fila)
                for clave, valor in bloque.get("resumen", {}).get(sensor, {}).items():
                    if clave in resumen[sensor]:
                        resumen[sensor][clave] += int(valor)
                estados[sensor].append(bloque.get("estado_sensores", {}).get(sensor, "desconocido"))
            procesados = n
            print(f"[HISTÓRICO] Bloque {n}/{total_bloques}: {desde} a {hasta}")

        for sensor, tmp in temporales.items():
            os.replace(tmp, nombres[sensor])
        estados_finales = {}
        for sensor, lista in estados.items():
            if all(e == "sin_cobertura_temporal" for e in lista):
                estados_finales[sensor] = "sin_cobertura_temporal"
            elif "cobertura_parcial" in lista or ("sin_cobertura_temporal" in lista and any(e in ("descarga_exitosa", "sin_detecciones") for e in lista)):
                estados_finales[sensor] = "cobertura_parcial"
            elif any(e == "descarga_exitosa" for e in lista):
                estados_finales[sensor] = "descarga_exitosa"
            elif all(e == "sin_detecciones" for e in lista):
                estados_finales[sensor] = "sin_detecciones"
            else:
                estados_finales[sensor] = lista[-1] if lista else "sin_disponibilidad"

        meta = {
            "bbox": list(bbox), "inicio": inicio.isoformat(), "fin": fin.isoformat(),
            "modo": "nasa_firms_historico_por_bloques", "archivos": nombres,
            "resumen": resumen, "estado_sensores": estados_finales,
            "bloques_completados": procesados, "bloques_totales": total_bloques,
        }
        manifest.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return meta
    except Exception:
        for p in temporales.values():
            p.unlink(missing_ok=True)
        raise
