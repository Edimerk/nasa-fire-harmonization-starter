
"""FireAtlas — Sprint 1: ingesta por BBOX desde NASA FIRMS.

Ejecución segura de prueba: python src/ingest.py --dry-run
Consulta real (requiere FIRMS_MAP_KEY): python src/ingest.py --run
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
from datetime import date, timedelta
from pathlib import Path

import httpx
from firms_client import BBox, fetch_area_range

BASE_DIR = Path(__file__).resolve().parent.parent
COLUMNAS = [
    "latitude", "longitude", "acq_date", "acq_time", "satellite", "instrument",
    "confidence", "type", "frp", "daynight", "source", "year", "timestamp",
    "brightness_k", "brightness_t5_k", "scan_km", "track_km", "version",
]
SENSORES = {
    "modis": "MODIS_NRT",
    "viirs_snpp": "VIIRS_SNPP_NRT",
    "viirs_noaa20": "VIIRS_NOAA20_NRT",
}


def validar_consulta(bbox, inicio: date, fin: date) -> tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise ValueError("El BBOX debe tener cuatro coordenadas (oeste, sur, este, norte).")
    oeste, sur, este, norte = (float(v) for v in bbox)
    if not (-180 <= oeste < este <= 180 and -90 <= sur < norte <= 90):
        raise ValueError("BBOX inválido o coordenadas fuera de rango.")
    if inicio > fin:
        raise ValueError("La fecha inicial debe ser anterior o igual a la final.")
    if fin > date.today():
        raise ValueError("La fecha final no puede estar en el futuro.")
    if (fin - inicio).days > 4:
        raise ValueError("Esta primera prueba permite un máximo de 5 días.")
    return (oeste, sur, este, norte)


def _fila(h, fuente: str) -> dict:
    fecha = getattr(h, "acq_date", None)
    return {
        "latitude": getattr(h, "latitude", None),
        "longitude": getattr(h, "longitude", None),
        "acq_date": fecha.isoformat() if isinstance(fecha, date) else fecha,
        "acq_time": getattr(h, "acq_time", None),
        "satellite": getattr(h, "satellite", None),
        "instrument": getattr(h, "instrument", None),
        "confidence": getattr(h, "confidence", None),
        "type": getattr(h, "detection_type", None),
        "frp": getattr(h, "frp_mw", None),
        "daynight": getattr(h, "daynight", None),
        "brightness_k": getattr(h, "brightness_k", None),
        "brightness_t5_k": getattr(h, "brightness_t5_k", None),
        "scan_km": getattr(h, "scan_km", None),
        "track_km": getattr(h, "track_km", None),
        "version": getattr(h, "version", None),
        "source": fuente,
        "year": fecha.year if isinstance(fecha, date) else None,
        "timestamp": (
            f"{fecha.isoformat()} "
            f"{int(getattr(h, 'acq_time')) // 100:02d}:"
            f"{int(getattr(h, 'acq_time')) % 100:02d}:00"
            if isinstance(fecha, date) and getattr(h, "acq_time", None) is not None
            else None
        ),
    }


def limpiar_filas(filas: list[dict], bbox, inicio: date, fin: date) -> tuple[list[dict], dict]:
    oeste, sur, este, norte = bbox
    salida = []
    rechazados = 0
    sin_tipo = 0

    for fila in filas:
        try:
            lat, lon = float(fila["latitude"]), float(fila["longitude"])
            fecha = date.fromisoformat(str(fila["acq_date"]))
            hora = int(fila["acq_time"])

            if not (sur <= lat <= norte and oeste <= lon <= este and inicio <= fecha <= fin):
                raise ValueError("Fuera de la consulta")

            if not (0 <= hora // 100 <= 23 and 0 <= hora % 100 <= 59):
                raise ValueError("Hora inválida")

            tipo = fila.get("type")

            if tipo not in (None, "", 0, "0"):
                rechazados += 1
                continue

            if tipo is None or tipo == "":
                sin_tipo += 1

            salida.append(fila)

        except (ValueError, TypeError, KeyError):
            rechazados += 1

    return salida, {
        "recibidos": len(filas),
        "limpios": len(salida),
        "descartados": rechazados,
        "sin_tipo": sin_tipo,
    }


def escribir_csv(ruta: Path, filas: list[dict]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS, extrasaction="ignore")
        escritor.writeheader()
        escritor.writerows(filas)


async def consultar_disponibilidad(client, map_key):
    url = (
        "https://firms.modaps.eosdis.nasa.gov/"
        f"api/data_availability/csv/{map_key}/all"
    )

    respuesta = await client.get(url, timeout=60)
    respuesta.raise_for_status()

    import io

    filas = csv.DictReader(io.StringIO(respuesta.text))

    return {
        fila["data_id"]: {
            "inicio": date.fromisoformat(fila["min_date"]),
            "fin": date.fromisoformat(fila["max_date"]),
        }
        for fila in filas
    }


async def _descargar(bbox, inicio, fin, map_key):
    resultados = {}
    estados = {}

    async with httpx.AsyncClient(timeout=120) as client:
        disponibilidad = await consultar_disponibilidad(client, map_key)

        for nombre, fuente in SENSORES.items():
            periodo = disponibilidad.get(fuente)

            if periodo is None:
                print(f"[SIN DISPONIBILIDAD] {fuente}")
                resultados[nombre] = []
                estados[nombre] = "sin_disponibilidad"
                continue

            fecha_inicio = max(inicio, periodo["inicio"])
            fecha_fin = min(fin, periodo["fin"])

            if fecha_inicio > fecha_fin:
                print(f"[SIN COBERTURA TEMPORAL] {fuente}")
                resultados[nombre] = []
                estados[nombre] = "sin_cobertura_temporal"
                continue

            if fecha_inicio != inicio or fecha_fin != fin:
                print(
                    f"[COBERTURA PARCIAL] {fuente}: "
                    f"{fecha_inicio} hasta {fecha_fin}"
                )

            print(f"[DESCARGANDO] {fuente}")

            registros = await fetch_area_range(
                client,
                map_key,
                fuente,
                BBox(bbox),
                fecha_inicio,
                fecha_fin,
            )

            resultados[nombre] = [
                _fila(h, fuente) for h in registros
            ]

            estados[nombre] = (
                "cobertura_parcial"
                if fecha_inicio != inicio or fecha_fin != fin
                else "descarga_exitosa"
                if resultados[nombre]
                else "sin_detecciones"
            )

            print(
                f"[RECIBIDOS] {fuente}: "
                f"{len(resultados[nombre])} registros"
            )

    return resultados, estados


def run_ingest(
    bbox,
    date_start: date,
    date_end: date,
    output_dir: Path,
    *,
    map_key: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Ejecuta una consulta acotada y devuelve las rutas de los CSV por sensor."""
    bbox = validar_consulta(bbox, date_start, date_end)
    salida = Path(output_dir)

    plan = {
        "bbox": list(bbox),
        "inicio": date_start.isoformat(),
        "fin": date_end.isoformat(),
        "fuentes": SENSORES,
        "modo": "simulacion" if dry_run else "nasa_firms",
    }

    if dry_run:
        return plan

    clave = map_key or os.environ.get("FIRMS_MAP_KEY", "")
    if not clave:
        raise RuntimeError("Falta FIRMS_MAP_KEY. Configure la clave antes de hacer una consulta real.")

    # Descargar datos y registrar estados de los sensores.
    datos, estados = asyncio.run(
        _descargar(bbox, date_start, date_end, clave)
    )

    # Eliminar CSV antiguos de sensores que ya no participan.
    for carpeta, sufijo in [
        ("raw", "_original.csv"),
        ("processed", "_filtrado.csv"),
    ]:
        directorio = salida / carpeta

        if directorio.exists():
            for archivo in directorio.glob(f"*{sufijo}"):
                nombre_sensor = archivo.name.removesuffix(sufijo)

                if nombre_sensor not in datos:
                    archivo.unlink()

    resumen = {}
    rutas = {}

    for nombre, filas in datos.items():
        limpios, conteos = limpiar_filas(filas, bbox, date_start, date_end)

        ruta_raw = salida / "raw" / f"{nombre}_original.csv"
        ruta_limpia = salida / "processed" / f"{nombre}_filtrado.csv"

        escribir_csv(ruta_raw, filas)
        escribir_csv(ruta_limpia, limpios)

        rutas[nombre] = str(ruta_limpia)
        resumen[nombre] = conteos

    plan.update({
        "archivos": rutas,
        "resumen": resumen,
        "estado_sensores": estados,
    })

    (salida / "consulta.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sprint 1 — NASA FIRMS por BBOX")
    modo = parser.add_mutually_exclusive_group(required=True)
    modo.add_argument("--dry-run", action="store_true", help="Comprueba sin descargar datos")
    modo.add_argument("--run", action="store_true", help="Consulta real (necesita FIRMS_MAP_KEY)")

    args = parser.parse_args()

    hoy = date.today()
    comienzo = hoy - timedelta(days=2)

    bbox_demo = (145.0, -38.0, 148.0, -35.0)
    clave_consulta = hashlib.sha256(
        f"{bbox_demo}|{comienzo}|{hoy}".encode()
    ).hexdigest()[:12]

    destino = BASE_DIR / "data" / "queries" / clave_consulta

    resultado = run_ingest(
        bbox_demo, comienzo, hoy, destino,
        dry_run=args.dry_run,
    )
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
