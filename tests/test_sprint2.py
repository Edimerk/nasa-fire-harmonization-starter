"""Pruebas Sprint 2. Colocar junto a otros archivos tests/ del proyecto."""
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from harmonize import normalize_confidence, run_harmonize, deduplicate


class Sprint2Tests(unittest.TestCase):
    def test_confianza_modis(self):
        self.assertEqual(normalize_confidence(20, "modis"), "low")
        self.assertEqual(normalize_confidence(54, "MODIS_NRT"), "nominal")
        self.assertEqual(normalize_confidence(90, "modis"), "high")

    def test_confianza_viirs(self):
        self.assertEqual(normalize_confidence("l", "viirs-snpp"), "low")
        self.assertEqual(normalize_confidence("n", "VIIRS_NOAA20_NRT"), "nominal")
        self.assertEqual(normalize_confidence("h", "viirs-snpp"), "high")

    def test_deduplicacion_prioridad_y_calidad(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            inp, out = base / "processed", base / "harmonized"
            inp.mkdir()
            pd.DataFrame([
                dict(latitude=3.32520, longitude=-71.96880, acq_date="2019-01-01", acq_time=347, source="modis", confidence=75, type=0, frp=40),
                dict(latitude=3.32525, longitude=-71.96885, acq_date="2019-01-01", acq_time=400, source="modis", confidence=15, type=0, frp=60),
            ]).to_csv(inp / "modis_filtrado.csv", index=False)
            pd.DataFrame([
                dict(latitude=3.32521, longitude=-71.96881, acq_date="2019-01-01", acq_time=400, source="VIIRS_SNPP_SP", confidence="n", type=0, frp=10),
                dict(latitude=3.50000, longitude=-72.00000, acq_date="2019-01-01", acq_time=400, source="VIIRS_SNPP_SP", confidence="h", type=0, frp=30),
            ]).to_csv(inp / "viirs_snpp_filtrado.csv", index=False)
            archivo = run_harmonize(inp, out)
            data = pd.read_csv(archivo)
            self.assertEqual(len(data), 2)
            self.assertTrue((data["source"] == "viirs-snpp").all())
            self.assertEqual(len(pd.read_csv(out / "duplicados_descartados.csv")), 1)
            self.assertTrue((out / "resumen_armonizacion.json").exists())
            self.assertTrue({"timestamp", "confidence_level", "priority", "year_month"}.issubset(data.columns))

    def test_viirs_vecinos_se_conservan(self):
        datos = pd.DataFrame([
            dict(latitude=-29.93331, longitude=151.03772, source="viirs-snpp", priority=0, frp=100, timestamp=pd.Timestamp("2019-02-15T03:19:00Z")),
            dict(latitude=-29.93665, longitude=151.04433, source="viirs-snpp", priority=0, frp=90, timestamp=pd.Timestamp("2019-02-15T03:19:00Z")),
        ])
        final, descartados = deduplicate(datos)
        self.assertEqual(len(final), 2)
        self.assertEqual(len(descartados), 0)

    def test_modis_viirs_coincidentes_se_unifican(self):
        datos = pd.DataFrame([
            dict(latitude=-29.93331, longitude=151.03772, source="viirs-snpp", priority=0, frp=100, timestamp=pd.Timestamp("2019-02-15T03:19:00Z")),
            dict(latitude=-29.93340, longitude=151.03780, source="modis", priority=1, frp=90, timestamp=pd.Timestamp("2019-02-15T03:20:00Z")),
        ])
        final, descartados = deduplicate(datos)
        self.assertEqual(len(final), 1)
        self.assertEqual(final.iloc[0]["source"], "viirs-snpp")
        self.assertEqual(descartados.iloc[0]["motivo_descarte"], "coincidencia_modis_viirs")

    def test_registros_exactos_mismo_sensor(self):
        datos = pd.DataFrame([
            dict(latitude=-29.93331, longitude=151.03772, source="viirs-snpp", priority=0, frp=100, timestamp=pd.Timestamp("2019-02-15T03:19:00Z")),
            dict(latitude=-29.93331, longitude=151.03772, source="viirs-snpp", priority=0, frp=100, timestamp=pd.Timestamp("2019-02-15T03:19:00Z")),
        ])
        final, descartados = deduplicate(datos)
        self.assertEqual(len(final), 1)
        self.assertEqual(descartados.iloc[0]["motivo_descarte"], "repeticion_exacta_mismo_sensor")

    def test_ausencia_archivos(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                run_harmonize(Path(tmp), Path(tmp) / "out")


if __name__ == "__main__":
    unittest.main()
