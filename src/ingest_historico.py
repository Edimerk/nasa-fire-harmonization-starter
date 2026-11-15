
"""FireAtlas - Sprint 1: ingesta histórica mundial por BBOX."""

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
from dataclasses import asdict
from datetime import date
from pathlib import Path

import httpx
from firms_client import BBox, fetch_area_range
from ingest import (
    BASE_DIR, _fila, limpiar_filas,
    escribir_csv, consultar_disponibilidad,
)

FUENTES = {
    "modis": "MODIS_SP",
    "viirs_snpp": "VIIRS_SNPP_SP",
}


async def descargar(bbox, inicio, fin, clave):
    datos = {}
    estados = {}

    async with httpx.AsyncClient(timeout=120) as client:
        disponibilidad = await consultar_disponibilidad(client, clave)

        for nombre, fuente in FUENTES.items():
            cobertura = disponibilidad.get(fuente)

            if cobertura is None:
                datos[nombre] = []
                estados[nombre] = "sin_disponibilidad"
                continue

            desde = max(inicio, cobertura["inicio"])
            hasta = min(fin, cobertura["fin"])

            if desde > hasta:
                datos[nombre] = []
                estados[nombre] = "sin_cobertura_temporal"
                continue

            print(f"[DESCARGANDO] {fuente}: {desde} a {hasta}")

            registros = await fetch_area_range(
                client, clave, fuente, BBox(bbox), desde, hasta
            )

            datos[nombre] = registros
            estados[nombre] = (
                "cobertura_parcial"
                if desde != inicio or hasta != fin
                else "descarga_exitosa"
                if registros else "sin_detecciones"
            )

            print(f"[RECIBIDOS] {fuente}: {len(registros)}")

    return datos, estados


def ejecutar(bbox, inicio, fin):
    bbox = tuple(map(float, bbox))

    if len(bbox) != 4 or not all(math.isfinite(v) for v in bbox):
        raise ValueError("BBOX inválido.")

    oeste, sur, este, norte = bbox

    if not (-180 <= oeste < este <= 180 and -90 <= sur < norte <= 90):
        raise ValueError("Coordenadas fuera de rango.")

    if inicio > fin or fin > date.today():
        raise ValueError("Fechas inválidas.")

    if (fin - inicio).days > 30:
        raise ValueError("Máximo 31 días por ejecución de prueba.")

    clave = os.getenv("FIRMS_MAP_KEY")
    if not clave:
        raise RuntimeError("Falta FIRMS_MAP_KEY.")

    identificador = hashlib.sha256(
        f"{bbox}|{inicio}|{fin}|historico".encode()
    ).hexdigest()[:12]

    salida = BASE_DIR / "data" / "queries" / identificador
    datos, estados = asyncio.run(descargar(bbox, inicio, fin, clave))

    resumen = {}
    rutas = {}

    for nombre, registros in datos.items():
        fuente = FUENTES[nombre]

        # Conservar todos los campos entregados por firms_client.
        originales = [asdict(h) for h in registros]

        ruta_raw = salida / "raw" / f"{nombre}_api.csv"
        ruta_raw.parent.mkdir(parents=True, exist_ok=True)

        if originales:
            with ruta_raw.open("w", newline="", encoding="utf-8") as f:
                escritor = csv.DictWriter(f, fieldnames=originales[0].keys())
                escritor.writeheader()
                for fila in originales:
                    fila["acq_date"] = fila["acq_date"].isoformat()
                    escritor.writerow(fila)

        filas = [_fila(h, fuente) for h in registros]
        limpios, conteos = limpiar_filas(filas, bbox, inicio, fin)

        ruta_procesada = salida / "processed" / f"{nombre}_filtrado.csv"
        escribir_csv(ruta_procesada, limpios)

        resumen[nombre] = conteos
        rutas[nombre] = str(ruta_procesada)

        print(f"[VEGETACIÓN VÁLIDA] {nombre}: {len(limpios)}")

    resultado = {
        "bbox": list(bbox),
        "inicio": inicio.isoformat(),
        "fin": fin.isoformat(),
        "fuentes": FUENTES,
        "modo": "nasa_firms_historico",
        "estado_sensores": estados,
        "resumen": resumen,
        "archivos": rutas,
    }

    salida.mkdir(parents=True, exist_ok=True)
    (salida / "consulta.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(resultado, indent=2, ensure_ascii=False))
    return resultado


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", nargs=4, type=float, required=True)
    parser.add_argument("--inicio", type=date.fromisoformat, required=True)
    parser.add_argument("--fin", type=date.fromisoformat, required=True)
    args = parser.parse_args()

    ejecutar(args.bbox, args.inicio, args.fin)
