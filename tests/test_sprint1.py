"""Pruebas locales y sin conexion para el Sprint 1 de FireAtlas.

Ubicar en: tests/test_sprint1_fireatlas.py
Ejecutar desde la raiz: python -m unittest discover -s tests -p 'test_sprint1_fireatlas.py' -v
"""
import csv
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ingest import COLUMNAS, _fila, limpiar_filas, escribir_csv, validar_consulta


class Sprint1IngestaTests(unittest.TestCase):
    BBOX = (-72.1, 3.2, -71.8, 3.5)
    INICIO = date(2019, 1, 1)
    FIN = date(2019, 1, 3)

    def fila(self, **cambios):
        fila = {
            "latitude": 3.324, "longitude": -71.9631,
            "acq_date": "2019-01-01", "acq_time": 347,
            "type": 0, "source": "MODIS_SP", "year": 2019,
            "timestamp": "2019-01-01 03:47:00",
        }
        fila.update(cambios)
        return fila

    def limpiar(self, filas):
        return limpiar_filas(filas, self.BBOX, self.INICIO, self.FIN)

    def test_bbox_valido(self):
        self.assertEqual(validar_consulta(self.BBOX, self.INICIO, self.FIN), self.BBOX)

    def test_bbox_invertido_es_rechazado(self):
        with self.assertRaises(ValueError):
            validar_consulta((-71.8, 3.2, -72.1, 3.5), self.INICIO, self.FIN)

    def test_fecha_final_anterior_es_rechazada(self):
        with self.assertRaises(ValueError):
            validar_consulta(self.BBOX, self.FIN, self.INICIO)

    def test_descarta_tipo_no_vegetacion(self):
        limpios, resumen = self.limpiar([self.fila(type=0), self.fila(type=2)])
        self.assertEqual(len(limpios), 1)
        self.assertEqual(resumen["descartados"], 1)

    def test_descarta_coordenadas_fuera_del_bbox(self):
        limpios, resumen = self.limpiar([self.fila(latitude=50.0)])
        self.assertEqual(len(limpios), 0)
        self.assertEqual(resumen["descartados"], 1)

    def test_descarta_hora_invalida(self):
        limpios, resumen = self.limpiar([self.fila(acq_time=2461)])
        self.assertEqual(len(limpios), 0)
        self.assertEqual(resumen["descartados"], 1)

    def test_sin_tipo_se_conserva_y_se_registra(self):
        limpios, resumen = self.limpiar([self.fila(type=None)])
        self.assertEqual(len(limpios), 1)
        self.assertEqual(resumen["sin_tipo"], 1)

    def test_fecha_fuera_del_intervalo_es_rechazada(self):
        limpios, resumen = self.limpiar([self.fila(acq_date="2019-02-01")])
        self.assertEqual(len(limpios), 0)
        self.assertEqual(resumen["descartados"], 1)

    def test_csv_vacio_conserva_encabezados(self):
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "processed" / "modis_filtrado.csv"
            escribir_csv(ruta, [])
            with ruta.open(encoding="utf-8", newline="") as archivo:
                lector = csv.DictReader(archivo)
                self.assertEqual(lector.fieldnames, COLUMNAS)
                self.assertEqual(list(lector), [])

    def test_trazabilidad_y_timestamp(self):
        hotspot = SimpleNamespace(
            latitude=3.324, longitude=-71.9631,
            acq_date=self.INICIO, acq_time=347,
            detection_type=0, satellite="Terra", instrument="MODIS",
            confidence="54", frp_mw=21.1,
        )
        fila = _fila(hotspot, "MODIS_SP")
        self.assertEqual(fila["source"], "MODIS_SP")
        self.assertEqual(fila["year"], 2019)
        self.assertEqual(fila["timestamp"], "2019-01-01 03:47:00")


if __name__ == "__main__":
    unittest.main(verbosity=2)
