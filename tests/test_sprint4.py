"""
Pruebas automatizadas unitarias para el Sprint 4 (Lógica analítica, validaciones y exportación).
NASA Space Apps Challenge 2026 - Harmonization of MODIS and VIIRS Hot Spots.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Agregar src al path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analytics_core import (
    validate_coordinates,
    validate_dates,
    filter_bbox,
    filter_dates,
    monthly_counts,
    calendar_matrix,
    historical_median,
    compute_anomalies,
    sanitize_for_csv,
    MESES_ES,
    NOMBRES_MESES,
)


class TestSprint4Analytics(unittest.TestCase):

    def setUp(self):
        # Dataset sintético pequeño de prueba (controlado y representativo)
        records = []
        # Años 2015 a 2019 con actividad fija para validar la mediana
        # Enero: 2015=10, 2016=20, 2017=30, 2018=40, 2019=50 -> Mediana = 30
        # Febrero: 0 detecciones en 2015, 2016, 2017; 10 en 2018, 2019 -> Mediana de [0,0,0,10,10] = 0
        for y in range(2015, 2020):
            for _ in range(y * 10 - 20140):  # 10, 20, 30, 40, 50 registros
                records.append({
                    "latitude": 4.5, "longitude": -74.1,
                    "timestamp": pd.Timestamp(f"{y}-01-15 12:00:00"),
                    "source": "viirs-snpp", "confidence_level": "nominal", "frp": 1.2
                })
        # Febrero solo en 2018 y 2019
        for y in [2018, 2019]:
            for _ in range(10):
                records.append({
                    "latitude": 5.0, "longitude": -73.0,
                    "timestamp": pd.Timestamp(f"{y}-02-10 14:00:00"),
                    "source": "modis", "confidence_level": "high", "frp": 2.5
                })
        # Año 2020 (año a evaluar)
        # Enero 2020: 60 focos (vs mediana 30 -> ratio 2.0 -> ALTA)
        for _ in range(60):
            records.append({
                "latitude": 4.5, "longitude": -74.1,
                "timestamp": pd.Timestamp("2020-01-20 12:00:00"),
                "source": "viirs-snpp", "confidence_level": "nominal", "frp": 1.5
            })
        # Febrero 2020: 5 focos (vs mediana 0 -> ALTA con mediana 0)
        for _ in range(5):
            records.append({
                "latitude": 5.0, "longitude": -73.0,
                "timestamp": pd.Timestamp("2020-02-15 12:00:00"),
                "source": "modis", "confidence_level": "high", "frp": 1.1
            })

        self.df = pd.DataFrame(records)
        self.df["year"] = self.df["timestamp"].dt.year
        self.df["month"] = self.df["timestamp"].dt.month
        self.df["month_name"] = self.df["month"].map(MESES_ES)

    def test_validate_coordinates_valid(self):
        val, msg = validate_coordinates(lat_min=1.0, lat_max=10.0, lon_min=-75.0, lon_max=-70.0)
        self.assertTrue(val)
        self.assertEqual(msg, "")

    def test_validate_coordinates_invalid_bounds(self):
        # Latitud mayor a 90 o menor a -90
        val, msg = validate_coordinates(lat_min=-95.0, lat_max=10.0, lon_min=-75.0, lon_max=-70.0)
        self.assertFalse(val)
        self.assertIn("latitud", msg.lower())

        # Longitud mayor a 180 o menor a -180
        val, msg = validate_coordinates(lat_min=1.0, lat_max=10.0, lon_min=-75.0, lon_max=190.0)
        self.assertFalse(val)
        self.assertIn("longitud", msg.lower())

    def test_validate_coordinates_inverted(self):
        # lat_min >= lat_max
        val, msg = validate_coordinates(lat_min=12.0, lat_max=4.0, lon_min=-75.0, lon_max=-70.0)
        self.assertFalse(val)
        self.assertIn("menor que la latitud máxima", msg)

        # lon_min >= lon_max
        val, msg = validate_coordinates(lat_min=4.0, lat_max=12.0, lon_min=-70.0, lon_max=-75.0)
        self.assertFalse(val)
        self.assertIn("menor que la longitud máxima", msg)

    def test_validate_dates(self):
        # Válidas
        val, _ = validate_dates("2015-01-01", "2020-12-31")
        self.assertTrue(val)

        # Invertidas
        val, msg = validate_dates("2020-05-10", "2018-01-01")
        self.assertFalse(val)
        self.assertIn("invertido", msg)

        # Fuera de rango permitido
        val, msg = validate_dates("2022-01-01", "2023-01-01", max_date="2020-12-31")
        self.assertFalse(val)

    def test_filter_bbox(self):
        # Bbox que solo cubre latitud 4.5 (deja por fuera lat 5.0)
        filtered = filter_bbox(self.df, bbox=(-75.0, 4.0, -73.5, 4.8))
        self.assertTrue(all(filtered["latitude"] == 4.5))
        self.assertTrue(all(filtered["longitude"] == -74.1))

    def test_filter_dates(self):
        filtered = filter_dates(self.df, "2020-01-01", "2020-01-31")
        self.assertEqual(len(filtered), 60)
        self.assertTrue(all(filtered["year"] == 2020))
        self.assertTrue(all(filtered["month"] == 1))

    def test_calendar_matrix_structure(self):
        cal = calendar_matrix(self.df)
        # Todos los 12 meses deben estar presentes como columnas
        self.assertEqual(cal.columns.tolist(), NOMBRES_MESES)
        # Los años presentes deben incluir 2015..2020
        self.assertIn(2015, cal.index)
        self.assertIn(2020, cal.index)
        # En 2015 Enero debe ser 10, y los meses sin registros deben ser 0
        self.assertEqual(cal.loc[2015, "Ene"], 10)
        self.assertEqual(cal.loc[2015, "Mar"], 0)

    def test_historical_median_imputation(self):
        # Mediana de Enero debe ser 30 (de [10, 20, 30, 40, 50])
        med_df = historical_median(self.df, baseline_years=[2015, 2016, 2017, 2018, 2019])
        ene_med = med_df[med_df["month"] == 1]["median_historical"].iloc[0]
        self.assertEqual(ene_med, 30.0)

        # Mediana de Febrero debe ser 0.0 (de [0, 0, 0, 10, 10] imputando ceros para 2015-2017)
        feb_med = med_df[med_df["month"] == 2]["median_historical"].iloc[0]
        self.assertEqual(feb_med, 0.0)

    def test_compute_anomalies_and_median_zero_handling(self):
        # Calcular anomalías para todo el dataset con baseline 2015-2019
        med_df = historical_median(self.df, baseline_years=[2015, 2016, 2017, 2018, 2019])
        anom = compute_anomalies(self.df, median_df=med_df)

        # Enero 2020: 60 focos vs mediana 30 -> ratio 2.0 -> ALTA
        row_ene_2020 = anom[(anom["year"] == 2020) & (anom["month"] == 1)].iloc[0]
        self.assertEqual(row_ene_2020["count"], 60)
        self.assertEqual(row_ene_2020["median_historical"], 30.0)
        self.assertAlmostEqual(row_ene_2020["anomaly_ratio"], 2.0)
        self.assertEqual(row_ene_2020["anomaly_label"], "alta")

        # Febrero 2020: 5 focos vs mediana 0 -> No debe crashear con ZeroDivisionError
        row_feb_2020 = anom[(anom["year"] == 2020) & (anom["month"] == 2)].iloc[0]
        self.assertEqual(row_feb_2020["count"], 5)
        self.assertEqual(row_feb_2020["median_historical"], 0.0)
        self.assertIn("alta", row_feb_2020["anomaly_label"])

    def test_sanitize_for_csv_injection(self):
        # Probar sanitización contra fórmulas maliciosas de hojas de cálculo
        test_df = pd.DataFrame({
            "source": ["=cmd|' /C calc'!A0", "+1234", "-danger", "@SUM(A1:A10)", "viirs-snpp"],
            "frp": [10.5, 20.0, 5.2, 8.1, 4.0]
        })
        sanitized = sanitize_for_csv(test_df)
        self.assertTrue(sanitized["source"].iloc[0].startswith("'="))
        self.assertTrue(sanitized["source"].iloc[1].startswith("'+"))
        self.assertTrue(sanitized["source"].iloc[2].startswith("'-"))
        self.assertTrue(sanitized["source"].iloc[3].startswith("'@"))
        self.assertEqual(sanitized["source"].iloc[4], "viirs-snpp")


if __name__ == "__main__":
    unittest.main()
