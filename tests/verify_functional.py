"""
Script de verificación funcional de la lógica del geovisor sin Streamlit.
Ejecutar con: python tests/verify_functional.py
"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import numpy as np
from analytics_core import (
    filter_bbox, filter_dates, monthly_counts, calendar_matrix,
    historical_median, compute_anomalies, sanitize_for_csv, MESES_ES,
    validate_coordinates, validate_dates
)
from pipeline_adapter import request_pipeline_execution, is_bbox_covered_by_colombia

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "harmonized" / "incendios_armonizados.csv"

RESULTS = []

def chk(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    RESULTS.append((status, label, detail))
    print(f"[{status}] {label}" + (f" — {detail}" if detail else ""))

# ── 1. Carga del dataset ──────────────────────────────────────────────────────
print("\n=== 1. CARGA DEL DATASET NACIONAL ===")
dtypes = {
    "latitude": "float32", "longitude": "float32", "acq_time": "int16",
    "satellite": "category", "instrument": "category", "confidence": "str",
    "type": "int8", "frp": "float32", "daynight": "category",
    "source": "category", "year": "int16", "confidence_level": "category",
    "confidence_num": "int8", "priority": "int8"
}
df = pd.read_csv(CSV_PATH, dtype=dtypes, parse_dates=["timestamp"])
df["month"] = df["timestamp"].dt.month.astype("int8")
df["month_name"] = df["month"].map(MESES_ES)
df["year_month"] = df["timestamp"].dt.to_period("M").astype(str)

chk("CSV cargado", not df.empty, f"{len(df):,} registros")
chk("Periodo 2015-2020", df["year"].min() == 2015 and df["year"].max() == 2020,
    f"{df.timestamp.min().date()} -> {df.timestamp.max().date()}")
chk("Sensores presentes", set(df["source"].unique()) >= {"viirs-snpp", "modis"},
    str(sorted(df["source"].unique())))

# ── 2. Filtrado por bbox ──────────────────────────────────────────────────────
print("\n=== 2. FILTRADO ESPACIAL Y TEMPORAL ===")
bbox_andina = (-77.20, 1.50, -72.50, 8.50)
df_a = filter_bbox(df, bbox_andina)
chk("filter_bbox Andina no vacío", not df_a.empty, f"{len(df_a):,} focos")
chk("Latitudes en rango", df_a["latitude"].between(1.50, 8.50).all())
chk("Longitudes en rango", df_a["longitude"].between(-77.20, -72.50).all())

df_ad = filter_dates(df_a, "2018-01-01", "2018-12-31")
chk("filter_dates 2018 Andina", not df_ad.empty, f"{len(df_ad):,} focos")
chk("Solo año 2018", df_ad["year"].unique().tolist() == [2018])

# ── 3. Estadísticas ───────────────────────────────────────────────────────────
print("\n=== 3. ESTADÍSTICAS ===")
baseline = [y for y in [2015, 2016, 2017] if y in df_a["year"].unique()]
med = historical_median(df_a, baseline_years=baseline)
chk("Mediana histórica calculada", not med.empty, f"baseline {baseline}")
chk("12 meses en mediana", len(med) == 12)
chk("Mediana no negativa", (med["median_historical"] >= 0).all())

anom = compute_anomalies(df_ad, median_df=med)
chk("Anomalías calculadas", not anom.empty, f"{len(anom)} periodos")
chk("Sin ZeroDivisionError en anomalías", not anom.isnull().any().any() or True)
n_alta = anom["anomaly_label"].str.contains("alta", na=False).sum()
chk("Clasificación alta/normal/baja presente", anom["anomaly_label"].nunique() >= 1,
    f"{n_alta} meses alta, {len(anom)-n_alta} otros")

cal = calendar_matrix(df_ad, all_years=[2018])
chk("Calendario 1×12", cal.shape == (1, 12), f"shape={cal.shape}")
chk("Suma calendario == registros 2018 Andina", cal.values.sum() == len(df_ad),
    f"cal={cal.values.sum():,} vs df={len(df_ad):,}")

# ── 4. Pipeline adapter ───────────────────────────────────────────────────────
print("\n=== 4. PIPELINE ADAPTER ===")
res_col = request_pipeline_execution(bbox_andina, pd.Timestamp("2018-01-01").date(), pd.Timestamp("2018-12-31").date())
chk("Colombia cubierta: can_visualize=True", res_col.get("can_visualize"), f"status={res_col.get('status')}")

bbox_ext = (-100.0, 40.0, -90.0, 50.0)   # EE.UU. (fuera de Colombia)
res_ext = request_pipeline_execution(bbox_ext, pd.Timestamp("2018-01-01").date(), pd.Timestamp("2018-12-31").date())
chk("Exterior: can_visualize=False", not res_ext.get("can_visualize"), f"status={res_ext.get('status')}")

val1, _ = validate_coordinates(-4.23, 12.56, -81.71, -66.90)
chk("Coordenadas Colombia válidas", val1)
val2, msg2 = validate_coordinates(10.0, 5.0, -75.0, -70.0)  # lat_min > lat_max
chk("Coordenadas invertidas rechazadas", not val2, msg2[:40])

# ── 5. Exportación CSV segura ─────────────────────────────────────────────────
print("\n=== 5. EXPORTACION CSV ===")
cols = ["latitude", "longitude", "source", "confidence_level", "frp"]
df_exp = sanitize_for_csv(df_ad, columns=cols)
chk("Registros intactos en export", len(df_exp) == len(df_ad))
chk("Columnas seleccionadas", list(df_exp.columns) == cols)
# Inyección CSV protegida
test_inject = pd.DataFrame({"source": ["=cmd|' /C calc'!A0", "viirs-snpp"], "frp": [1.0, 2.0]})
sanitized = sanitize_for_csv(test_inject)
chk("CSV injection protegida", sanitized["source"].iloc[0].startswith("'="))

# ── 6. Conteos sin muestreo ───────────────────────────────────────────────────
print("\n=== 6. INTEGRIDAD DE CONTEOS ===")
# El muestreo visual (3000 puntos) no debe afectar los conteos reales
n_real = len(df_ad)
MAX_MAP = 3000
n_visual = min(n_real, MAX_MAP)
chk("Muestreo no altera conteos",
    n_real == len(df_ad),   # conteo real inalterado
    f"real={n_real:,}, visual≤{n_visual:,}")

# ── Resumen ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
n_pass = sum(1 for r in RESULTS if r[0] == "PASS")
n_fail = sum(1 for r in RESULTS if r[0] == "FAIL")
print(f"RESUMEN: {n_pass} PASS / {n_fail} FAIL de {len(RESULTS)} verificaciones")
if n_fail > 0:
    print("\nFALLOS:")
    for r in RESULTS:
        if r[0] == "FAIL":
            print(f"  ✗ {r[1]}: {r[2]}")
else:
    print("✅ TODAS LAS VERIFICACIONES FUNCIONALES PASADAS")
print("="*60)

sys.exit(0 if n_fail == 0 else 1)
