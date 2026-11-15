"""Sprint 2: armonización MODIS/VIIRS (datos FIRMS ya filtrados en Sprint 1).

Uso de prueba:
  python src/harmonize.py --processed-dir data/queries/294f8ce02ea7/processed --output-dir data/queries/294f8ce02ea7/harmonized

No sobrescribe la base histórica precalculada a menos que se indique esa ruta explícitamente.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

RADIO_M = 1000
VENTANA_HORAS = 3
RADIO_TIERRA_M = 6_371_000
MESES_ES = {1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"}


def normalize_confidence(confianza, fuente):
    """Criterios del proyecto original: MODIS <30 low, <80 nominal, >=80 high;
    VIIRS l/n/h. Valores desconocidos se registran como nominal (provisional)."""
    valor = str(confianza).strip().lower()
    fuente = str(fuente).lower()
    if "modis" in fuente:
        try:
            porcentaje = float(valor)
            if not 0 <= porcentaje <= 100 or not np.isfinite(porcentaje):
                return "unknown"
            return "low" if porcentaje < 30 else ("nominal" if porcentaje < 80 else "high")
        except (ValueError, TypeError):
            return "unknown"
    return {"l": "low", "low": "low", "n": "nominal", "nominal": "nominal",
            "h": "high", "high": "high"}.get(valor, "unknown")


def _normalizar(df: pd.DataFrame, archivo: Path) -> pd.DataFrame:
    df = df.copy()
    for campo in ("latitude", "longitude", "source", "confidence"):
        if campo not in df.columns:
            raise ValueError(f"{archivo.name}: falta columna requerida '{campo}'")
    if "timestamp" not in df.columns:
        if not {"acq_date", "acq_time"}.issubset(df.columns):
            raise ValueError(f"{archivo.name}: falta timestamp o fecha/hora")
        horas = pd.to_numeric(df["acq_time"], errors="coerce")
        hhmm = horas.fillna(-1).astype(int).astype(str).str.zfill(4)
        df["timestamp"] = df["acq_date"].astype(str) + " " + hhmm.str[:2] + ":" + hhmm.str[2:4]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["frp"] = pd.to_numeric(df.get("frp", pd.Series(np.nan, index=df.index)), errors="coerce")
    df["source_raw"] = df["source"].astype(str)
    fuentes = df["source_raw"].str.lower()
    df["source"] = np.select(
        [fuentes.str.contains("modis"), fuentes.str.contains("noaa20"),
         fuentes.str.contains("noaa21"), fuentes.str.contains("viirs")],
        ["modis", "viirs-noaa20", "viirs-noaa21", "viirs-snpp"], default=df["source_raw"]
    )
    tipo = pd.to_numeric(df.get("type", pd.Series(np.nan, index=df.index)), errors="coerce")
    df["type"] = tipo
    df["type_unverified"] = tipo.isna()
    ok = (df["latitude"].between(-90, 90) & df["longitude"].between(-180, 180)
          & df["timestamp"].notna() & (tipo.eq(0) | tipo.isna()))
    df = df.loc[ok].copy()
    df["confidence_level"] = [normalize_confidence(c, s) for c, s in zip(df["confidence"], df["source"])]
    df["confidence_num"] = df["confidence_level"].map({"unknown": -1, "low": 0, "nominal": 1, "high": 2}).astype(int)
    df["priority"] = np.where(df["source"].str.contains("viirs"), 0, 1)
    df["year"] = df["timestamp"].dt.year.astype(int)
    df["month"] = df["timestamp"].dt.month.astype(int)
    df["month_name"] = df["month"].map(MESES_ES)
    df["year_month"] = df["timestamp"].dt.strftime("%Y-%m")
    df["input_file"] = archivo.name
    return df


def _xyz(df: pd.DataFrame) -> np.ndarray:
    lat = np.radians(df["latitude"].to_numpy(dtype=float))
    lon = np.radians(df["longitude"].to_numpy(dtype=float))
    return np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat))) * RADIO_TIERRA_M


def deduplicate(df: pd.DataFrame, radio_m: float = RADIO_M, ventana_horas: float = VENTANA_HORAS):
    """Conserva píxeles vecinos del mismo sensor; revisa coincidencias MODIS/VIIRS.

    - Una fila repetida del mismo producto, con iguales fuente, coordenadas y
      timestamp, se elimina como duplicado exacto.
    - La regla radio/tiempo se usa SOLO entre familias MODIS y VIIRS. Se
      conserva VIIRS en la coincidencia (preferencia de resolución).
    - Dos observaciones VIIRS vecinas no son necesariamente el mismo píxel.
    - Esta es una heurística de superposición, no identifica incendios únicos.
    """
    if df.empty:
        return df.copy(), df.copy()
    orden = df.sort_values(
        ["priority", "frp", "timestamp"],
        ascending=[True, False, True], na_position="last"
    ).reset_index(drop=True)
    puntos = _xyz(orden)
    tree = cKDTree(puntos)
    tiempo = orden["timestamp"].astype("int64").to_numpy() / 1e9
    fuentes = orden["source"].astype(str).str.lower().to_numpy()
    familias = np.where(np.char.find(fuentes.astype(str), "viirs") >= 0, "viirs",
                        np.where(np.char.find(fuentes.astype(str), "modis") >= 0,
                                 "modis", "otro"))
    ventana_seg = ventana_horas * 3600
    radio_cuerda = 2 * RADIO_TIERRA_M * np.sin(radio_m / (2 * RADIO_TIERRA_M))
    keep = np.ones(len(orden), dtype=bool)
    eliminado_por = np.full(len(orden), -1, dtype=np.int64)
    motivos = np.full(len(orden), "", dtype=object)
    vistos_exactos = {}
    for i in range(len(orden)):
        clave = (fuentes[i], float(orden.at[i, "latitude"]),
                 float(orden.at[i, "longitude"]), int(orden.at[i, "timestamp"].value))
        if clave in vistos_exactos:
            keep[i] = False
            eliminado_por[i] = vistos_exactos[clave]
            motivos[i] = "repeticion_exacta_mismo_sensor"
            continue
        vistos_exactos[clave] = i
        # VIIRS precede a MODIS; solo se descarta MODIS si coincide con VIIRS.
        if familias[i] != "modis":
            continue
        vecinos = tree.query_ball_point(puntos[i], r=radio_cuerda)
        candidatos = [j for j in vecinos if keep[j] and familias[j] == "viirs"
                      and abs(tiempo[j] - tiempo[i]) <= ventana_seg]
        if candidatos:
            j = min(candidatos, key=lambda k: (abs(tiempo[k] - tiempo[i]),
                                               float(np.linalg.norm(puntos[k] - puntos[i]))))
            keep[i] = False
            eliminado_por[i] = j
            motivos[i] = "coincidencia_modis_viirs"
    conservados = orden.loc[keep].copy()
    descartados = orden.loc[~keep].copy()
    descartados["duplicado_de_indice"] = eliminado_por[~keep]
    descartados["duplicado_de_fuente"] = [orden.at[i, "source"] for i in eliminado_por[~keep]]
    descartados["motivo_descarte"] = motivos[~keep]
    return conservados.reset_index(drop=True), descartados.reset_index(drop=True)


def run_harmonize(processed_dir: Path, output_dir: Path, bbox: tuple | None = None) -> Path:
    """Contrato Sprint 2: CSV Sprint 1 -> CSV armonizado + informe y auditoría."""
    origen, salida = Path(processed_dir), Path(output_dir)
    archivos = sorted(origen.glob("*_filtrado.csv"))
    if not archivos:
        raise FileNotFoundError(f"No hay archivos *_filtrado.csv en {origen}")
    partes = []
    for archivo in archivos:
        datos = pd.read_csv(archivo, low_memory=False)
        if datos.empty:
            continue
        partes.append(_normalizar(datos, archivo))
    if not partes:
        raise ValueError("Los CSV existen pero no contienen detecciones")
    df = pd.concat(partes, ignore_index=True)
    entradas_validas = len(df)
    if bbox is not None:
        oeste, sur, este, norte = map(float, bbox)
        df = df.loc[df.latitude.between(sur, norte) & df.longitude.between(oeste, este)].copy()
    sin_tipo = int(df["type_unverified"].sum())
    desconocida = int((df["confidence_level"] == "unknown").sum())
    low = int((df["confidence_level"] == "low").sum())
    # La baja confianza se excluye; confianza desconocida se preserva y se registra.
    df = df.loc[df["confidence_level"] != "low"].copy()
    maestro, descartados = deduplicate(df)
    salida.mkdir(parents=True, exist_ok=True)
    final = salida / "incendios_armonizados.csv"
    maestro.to_csv(final, index=False, encoding="utf-8")
    descartados.to_csv(salida / "duplicados_descartados.csv", index=False, encoding="utf-8")
    informe = {
        "archivos_entrada": [str(p) for p in archivos],
        "registros_validos_antes_bbox": entradas_validas,
        "registros_tras_bbox": int(len(df) + low),
        "confianza_baja_descartados": low,
        "confianza_desconocida_antes_filtro": desconocida,
        "tipo_no_verificable_antes_filtro": sin_tipo,
        "duplicados_espaciotemporales": int((descartados["motivo_descarte"] == "coincidencia_modis_viirs").sum()) if not descartados.empty else 0,
        "repeticiones_exactas": int((descartados["motivo_descarte"] == "repeticion_exacta_mismo_sensor").sum()) if not descartados.empty else 0,
        "registros_descartados_deduplicacion": int(len(descartados)),
        "registros_armonizados": int(len(maestro)),
        "radio_m": RADIO_M, "ventana_horas": VENTANA_HORAS,
        "prioridad": "VIIRS", "timestamp_timezone": "UTC",
        "politica_deduplicacion": "Solo solapamiento MODIS-VIIRS por radio/tiempo, y repetición exacta dentro del mismo sensor; píxeles vecinos se conservan",
        "nota": "Detecciones con type ausente se conservan como no verificadas; no equivalen a incendios confirmados."
    }
    (salida / "resumen_armonizacion.json").write_text(json.dumps(informe, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(informe, indent=2, ensure_ascii=False))
    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sprint 2: armonización MODIS + VIIRS")
    parser.add_argument("--processed-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print("CSV ARMONIZADO:", run_harmonize(args.processed_dir, args.output_dir))
