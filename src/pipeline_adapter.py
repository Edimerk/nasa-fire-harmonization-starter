"""
Adaptador de integración del Pipeline de Datos (Sprints 1 → 2 → 3 → 4).
NASA Space Apps Challenge 2026 - Harmonization of MODIS and VIIRS Hot Spots.

Este módulo implementa el contrato de integración solicitado por la profesora:
- Conecta el visor geográfico (Sprint 4) con la ingesta (Sprint 1), armonización (Sprint 2)
  y analítica (Sprint 3).
- Verifica la disponibilidad, integridad y signatura de los módulos de otros integrantes.
- Evita descargas o cómputos redundantes si los datos ya fueron procesados o pertenecen
  a la cobertura precalculada.
- Documenta el contrato exacto de parámetros e interfaces que deben cumplir los compañeros.
- Maneja errores de forma segura sin simular falsos positivos ni alterar los datos existentes.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import datetime
import importlib
import logging
import hashlib
import pandas as pd

from analytics_core import validate_coordinates, validate_dates

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
HARMONIZED_DIR = DATA_DIR / "harmonize"
ANALYTICS_DIR = DATA_DIR / "analytics"
QUERIES_CACHE_DIR = DATA_DIR / "queries_cache"

# Bounding box de la cobertura nacional precalculada (Colombia 2015-2020)
BBOX_COLOMBIA_COBERTURA = (-81.71, -4.23, -66.90, 12.56)
FECHA_MIN_COLOMBIA = datetime.date(2015, 1, 1)
FECHA_MAX_COLOMBIA = datetime.date(2020, 12, 31)


def is_bbox_covered_by_colombia(bbox: Tuple[float, float, float, float]) -> bool:
    """
    Comprueba si un Bounding Box solicitado (w, s, e, n) está completamente contenido
    dentro de la cobertura precalculada de Colombia 2015-2020.
    """
    w, s, e, n = bbox
    cw, cs, ce, cn = BBOX_COLOMBIA_COBERTURA
    return (w >= cw - 0.2) and (e <= ce + 0.2) and (s >= cs - 0.2) and (n <= cn + 0.2)


def get_query_hash(bbox: Tuple[float, float, float, float], date_start: datetime.date, date_end: datetime.date) -> str:
    """Genera un hash determinista para identificar de forma unívoca una consulta geográfica y temporal."""
    key = f"{bbox[0]:.2f}_{bbox[1]:.2f}_{bbox[2]:.2f}_{bbox[3]:.2f}_{date_start}_{date_end}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:12]


def check_pipeline_modules() -> Dict[str, Dict[str, Any]]:
    """
    Inspecciona los archivos fuente de los Sprints 1, 2 y 3 en src/
    determinando si existen, si están vacíos (0 bytes) y si exponen las interfaces requeridas.
    """
    modules_info = {
        "config": {
            "expected_file": "config.py",
            "path": SRC_DIR / "config.py",
            "status": "missing",
            "size_bytes": 0,
            "has_required_interface": False,
            "details": ""
        },
        "ingest": {
            "expected_file": "ingest.py",
            "path": SRC_DIR / "ingest.py",
            "alias_path": SRC_DIR / "ingest.py",
            "status": "missing",
            "size_bytes": 0,
            "has_required_interface": False,
            "details": ""
        },
        "harmonize": {
            "expected_file": "harmonize.py",
            "path": SRC_DIR / "harmonize.py",
            "alias_path": SRC_DIR / "harmonized.py",
            "status": "missing",
            "size_bytes": 0,
            "has_required_interface": False,
            "details": ""
        },
        "analytics": {
            "expected_file": "analytics.py",
            "path": SRC_DIR / "analytics.py",
            "status": "missing",
            "size_bytes": 0,
            "has_required_interface": False,
            "details": ""
        }
    }

    # 1. Config
    p_cfg = modules_info["config"]["path"]
    if p_cfg.exists():
        size = p_cfg.stat().st_size
        modules_info["config"]["size_bytes"] = size
        if size == 0:
            modules_info["config"]["status"] = "empty"
            modules_info["config"]["details"] = "El archivo existe pero está vacío (0 bytes)."
        else:
            modules_info["config"]["status"] = "present"
            modules_info["config"]["has_required_interface"] = True
            modules_info["config"]["details"] = "Módulo config.py cargado."
    else:
        modules_info["config"]["details"] = "Archivo config.py no encontrado."

    # 2. Ingest (Sprint 1)
    p_ing = modules_info["ingest"]["path"]
    p_ing_alias = modules_info["ingest"]["alias_path"]
    target_ing = p_ing if p_ing.exists() else (p_ing_alias if p_ing_alias.exists() else None)

    if target_ing:
        size = target_ing.stat().st_size
        modules_info["ingest"]["size_bytes"] = size
        modules_info["ingest"]["path"] = target_ing
        if size == 0:
            modules_info["ingest"]["status"] = "empty"
            modules_info["ingest"]["details"] = f"El archivo {target_ing.name} existe pero está vacío (0 bytes)."
        else:
            modules_info["ingest"]["status"] = "present"
            # Verificar si tiene función run_ingest o similar
            try:
                content = target_ing.read_text(encoding="utf-8", errors="ignore")
                if "def run_ingest" in content or "def descargar_sensor" in content:
                    modules_info["ingest"]["has_required_interface"] = "def run_ingest" in content
                    modules_info["ingest"]["details"] = "Implementación encontrada en código fuente."
                else:
                    modules_info["ingest"]["details"] = "Implementación presente pero sin interfaz parametrizada por BBox."
            except Exception as e:
                modules_info["ingest"]["details"] = f"Error leyendo archivo: {e}"
    else:
        modules_info["ingest"]["details"] = "Script de ingesta (ingest.py) no encontrado en src/."

    # 3. Harmonize (Sprint 2)
    p_harm = modules_info["harmonize"]["path"]
    p_harm_alias = modules_info["harmonize"]["alias_path"]
    target_harm = p_harm if p_harm.exists() else (p_harm_alias if p_harm_alias.exists() else None)

    if target_harm:
        size = target_harm.stat().st_size
        modules_info["harmonize"]["size_bytes"] = size
        modules_info["harmonize"]["path"] = target_harm
        if size == 0:
            modules_info["harmonize"]["status"] = "empty"
            modules_info["harmonize"]["details"] = f"El archivo {target_harm.name} existe pero está vacío (0 bytes)."
        else:
            modules_info["harmonize"]["status"] = "present"
            try:
                content = target_harm.read_text(encoding="utf-8", errors="ignore")
                modules_info["harmonize"]["has_required_interface"] = "def run_harmonize" in content
                modules_info["harmonize"]["details"] = "Implementación encontrada en código fuente."
            except Exception as e:
                modules_info["harmonize"]["details"] = f"Error leyendo archivo: {e}"
    else:
        modules_info["harmonize"]["details"] = "Script de armonización (harmonize.py) no encontrado en src/."

    # 4. Analytics (Sprint 3)
    p_ana = modules_info["analytics"]["path"]
    if p_ana.exists():
        size = p_ana.stat().st_size
        modules_info["analytics"]["size_bytes"] = size
        if size == 0:
            modules_info["analytics"]["status"] = "empty"
            modules_info["analytics"]["details"] = "El archivo analytics.py existe pero está vacío (0 bytes)."
        else:
            modules_info["analytics"]["status"] = "present"
            try:
                content = p_ana.read_text(encoding="utf-8", errors="ignore")
                modules_info["analytics"]["has_required_interface"] = "def run_analytics" in content or "def compute_anomalies" in content
                modules_info["analytics"]["details"] = "Implementación encontrada en código fuente."
            except Exception as e:
                modules_info["analytics"]["details"] = f"Error leyendo archivo: {e}"
    else:
        modules_info["analytics"]["details"] = "Script de analítica (analytics.py) no encontrado en src/."

    return modules_info


def get_required_contracts_doc() -> Dict[str, Any]:
    """
    Retorna el contrato técnico exacto que deben implementar los responsables
    de los Sprints 1, 2 y 3 para conectarse con la interfaz del Sprint 4.
    """
    return {
        "sprint_1_ingest": {
            "modulo": "src/ingest.py",
            "responsable": "Compañero Sprint 1 (Ingesta)",
            "firma_requerida": "run_ingest(bbox: tuple, date_start: date, date_end: date, output_dir: Path) -> Dict[str, Path]",
            "parametros": {
                "bbox": "Tupla (lon_min, lat_min, lon_max, lat_max) en grados WGS84.",
                "date_start": "datetime.date con la fecha inicial.",
                "date_end": "datetime.date con la fecha final.",
                "output_dir": "Path donde se escribirán los CSV filtrados."
            },
            "salidas_esperadas": [
                "output_dir / 'modis_filtrado.csv'",
                "output_dir / 'viirs_snpp_filtrado.csv'"
            ],
            "regla_cientifica": "Filtrar únicamente detecciones con type == 0 (vegetación) y preservar columnas de trazabilidad."
        },
        "sprint_2_harmonize": {
            "modulo": "src/harmonize.py",
            "responsable": "Compañero Sprint 2 (Armonización)",
            "firma_requerida": "run_harmonize(processed_dir: Path, output_dir: Path, bbox: tuple = None) -> Path",
            "parametros": {
                "processed_dir": "Directorio con los CSVs entregados por Sprint 1.",
                "output_dir": "Directorio donde se escribirá el dataset armonizado.",
                "bbox": "Opcional: Bounding Box para recorte si no fue aplicado en ingesta."
            },
            "salidas_esperadas": [
                "output_dir / 'incendios_armonizados.csv'"
            ],
            "regla_cientifica": "Normalización de confianza (low/nominal/high), deduplicación espaciotemporal (1 km, 3 h, prioridad VIIRS) y preservación de provenance."
        },
        "sprint_3_analytics": {
            "modulo": "src/analytics.py",
            "responsable": "Compañero Sprint 3 (Analítica)",
            "firma_requerida": "run_analytics(harmonized_csv_path: Path, output_dir: Path) -> Dict[str, Path]",
            "parametros": {
                "harmonized_csv_path": "Ruta hacia el archivo incendios_armonizados.csv generado en Sprint 2.",
                "output_dir": "Directorio donde se escribirán las tablas precalculadas."
            },
            "salidas_esperadas": [
                "output_dir / 'monthly_counts.csv'",
                "output_dir / 'annual_counts.csv'",
                "output_dir / 'anomalies.csv'",
                "output_dir / 'historical_median.csv'",
                "output_dir / 'calendar_matrix.csv'"
            ],
            "regla_cientifica": "Mediana histórica mensual excluyendo año actual, ratios de anomalía contra mediana (manejando mediana cero) y matriz de calendario 12 meses."
        }
    }


def request_pipeline_execution(
    bbox: Tuple[float, float, float, float],
    date_start: datetime.date,
    date_end: datetime.date
) -> Dict[str, Any]:
    """
    Punto de entrada de orquestación desde la interfaz de Streamlit.
    Evalúa el estado de la consulta:
    1. Si está en Colombia 2015-2020: Indica que se puede procesar directamente con los datos armonizados verificados.
    2. Si es una nueva región externa: Verifica si los módulos de Sprints 1, 2 y 3 están listos.
       - Si están vacíos o incompletos: Retorna el informe de bloqueo con los contratos técnicos detallados.
       - Si están implementados: Ejecuta la secuencia ingest -> harmonize -> analytics.
    """
    val_geo, msg_geo = validate_coordinates(bbox[1], bbox[3], bbox[0], bbox[2])
    if not val_geo:
        return {"success": False, "status": "invalid_coordinates", "message": msg_geo}

    val_dt, msg_dt = validate_dates(date_start, date_end)
    if not val_dt:
        return {"success": False, "status": "invalid_dates", "message": msg_dt}

    modules = check_pipeline_modules()

    # Comprobar si los módulos requeridos están operativos
    ingest_ready = modules["ingest"]["status"] == "present" and modules["ingest"]["has_required_interface"]
    harmonize_ready = modules["harmonize"]["status"] == "present" and modules["harmonize"]["has_required_interface"]
    analytics_ready = modules["analytics"]["status"] == "present" and modules["analytics"]["has_required_interface"]

    all_modules_ready = ingest_ready and harmonize_ready and analytics_ready

    # Caso 1: Región dentro del dataset nacional existente
    if is_bbox_covered_by_colombia(bbox) and (date_start >= FECHA_MIN_COLOMBIA) and (date_end <= FECHA_MAX_COLOMBIA):
        return {
            "success": True,
            "status": "cached_dataset_available",
            "source": "preprocessed_colombia_dataset",
            "message": "La región seleccionada está cubierta por el dataset armonizado verificado de Colombia (2015–2020). Los datos y la analítica se visualizan de inmediato.",
            "modules_status": modules,
            "can_visualize": True
        }

    # Caso 2: Nueva región que requiere el pipeline completo
    if not all_modules_ready:
        missing_list = []
        if not ingest_ready:
            missing_list.append(f"Sprint 1 (ingest.py: {modules['ingest']['details']})")
        if not harmonize_ready:
            missing_list.append(f"Sprint 2 (harmonize.py: {modules['harmonize']['details']})")
        if not analytics_ready:
            missing_list.append(f"Sprint 3 (analytics.py: {modules['analytics']['details']})")

        contracts = get_required_contracts_doc()

        return {
            "success": False,
            "status": "modules_not_ready",
            "source": "pipeline_required",
            "message": (
                "Para procesar un área geográfica fuera de la cobertura precalculada de Colombia, "
                "se requiere la ejecución secuencial de los Sprints 1, 2 y 3. Sin embargo, los módulos "
                "de los compañeros se encuentran pendientes de entrega o vacíos."
            ),
            "pending_modules": missing_list,
            "modules_status": modules,
            "contracts": contracts,
            "can_visualize": False
        }

    # Caso 3: Módulos listos (cuando los compañeros entreguen sus scripts)
    return {
        "success": False,
        "status": "execution_ready",
        "message": "Los módulos de Sprints 1, 2 y 3 están presentes y listos para ejecutarse.",
        "modules_status": modules,
        "can_visualize": False
    }


# Integración provisional Sprint 1: muestra detecciones SIN armonizar.
def request_ingest_preview(bbox, date_start, date_end):
    """NASA FIRMS: ingesta controlada, sin modificar los Sprints 2 ni 3.

    Consultas históricas de hasta 93 días mediante bloques cacheados de 5.
    Consultas recientes conservan el máximo inicial de 5 días.
    """
    val_geo, msg_geo = validate_coordinates(bbox[1], bbox[3], bbox[0], bbox[2])
    val_dt, msg_dt = validate_dates(date_start, date_end)
    if not val_geo or not val_dt:
        return {"success": False, "message": msg_geo if not val_geo else msg_dt}
    try:
        antiguo = date_end < datetime.date.today() - datetime.timedelta(days=120)
        dias = (date_end - date_start).days + 1
        if antiguo:
            from historical_batches import descargar_historico_por_bloques
            resultado = descargar_historico_por_bloques(tuple(bbox), date_start, date_end)
        else:
            if dias > 5:
                return {"success": False, "message": (
                    "Las consultas recientes o mixtas siguen limitadas a 5 días por ahora. "
                    "Para periodos históricos terminados hace más de 120 días, "
                    "FireTrackers admite hasta 93 días con caché por bloques.")}
            from ingest import run_ingest
            clave = get_query_hash(bbox, date_start, date_end)
            destino = DATA_DIR / "web_preview" / clave
            resultado = run_ingest(bbox, date_start, date_end, destino)
        return {
            "success": True,
            "status": "ingest_preview",
            "message": "Detecciones NASA FIRMS listas para armonizar (Sprint 1).",
            "archivos": resultado.get("archivos", {}),
            "estado_sensores": resultado.get("estado_sensores", {}),
            "resumen": resultado.get("resumen", {}),
        }
    except Exception as error:
        # No mostrar URLs ni claves dentro del error HTTP.
        logger.error("Error de consulta FIRMS: %s", type(error).__name__)
        return {"success": False, "message": (
            "No se completó la descarga FIRMS. Comprueba disponibilidad, tamaño "
            "del rango, conexión y clave del servidor. Detalle: "
            + str(error) if isinstance(error, ValueError) else
            "No se completó la descarga FIRMS. Revisa la disponibilidad "
            "y el registro de errores del servidor.")}


# Integración Sprints 1 -> 2 -> 3 para regiones externas a la base Colombia.
def request_full_pipeline(bbox, date_start, date_end):
    """Ingesta NASA -> armonización -> agregaciones. No estima anomalías sin baseline."""
    ingesta = request_ingest_preview(bbox, date_start, date_end)
    if not ingesta.get("success"):
        return ingesta

    try:
        from harmonize import run_harmonize
        from analytics import run as run_analytics

        archivos = [Path(ruta) for ruta in ingesta.get("archivos", {}).values() if ruta]
        existentes = [p for p in archivos if p.exists() and p.name.endswith("_filtrado.csv")]
        if not existentes:
            return {"success": True, "status": "no_detections", "message":
                    "NASA no devolvió detecciones procesadas para esta consulta.",
                    "estado_sensores": ingesta.get("estado_sensores", {})}

        directorios = {p.parent.resolve() for p in existentes}
        if len(directorios) != 1:
            raise ValueError("Los CSV procesados no comparten directorio de origen")
        processed_dir = existentes[0].parent
        query_dir = DATA_DIR / "queries" / get_query_hash(bbox, date_start, date_end)
        harmonized_dir = query_dir / "harmonized"
        analytics_dir = query_dir / "analytics"
        final_csv = run_harmonize(processed_dir, harmonized_dir, bbox=bbox)
        informe = run_analytics(final_csv, analytics_dir,
                                bbox=bbox, start=date_start.isoformat(),
                                end=date_end.isoformat())
        return {"success": True, "status": "pipeline_completed",
                "csv_path": str(final_csv), "analytics_dir": str(analytics_dir),
                "analytics_summary": informe,
                "harmonization_summary": str(harmonized_dir / "resumen_armonizacion.json"),
                "estado_sensores": ingesta.get("estado_sensores", {}),
                "message": "Registros armonizados y agregados correctamente."}
    except Exception as error:
        logger.exception("Error al procesar Sprints 2 y 3")
        return {"success": False, "status": "pipeline_error",
                "message": f"La consulta NASA se realizó, pero falló la armonización o analítica: {error}"}
