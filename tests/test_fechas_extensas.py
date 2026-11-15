"""Pruebas desconectadas de NASA de la nueva ingesta por bloques."""
import csv
import json
import sys
import tempfile
import types
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import historical_batches as hb


class FechasExtensas(unittest.TestCase):
    def test_bloques_no_solapan(self):
        inicio, fin = date(2017, 12, 29), date(2018, 1, 10)
        self.assertEqual(list(hb.dividir_intervalo(inicio, fin)), [
            (date(2017, 12, 29), date(2018, 1, 2)),
            (date(2018, 1, 3), date(2018, 1, 7)),
            (date(2018, 1, 8), date(2018, 1, 10)),
        ])

    def test_rechaza_demasiado_amplio(self):
        with self.assertRaisesRegex(ValueError, "93"):
            hb.descargar_historico_por_bloques((145, -38, 150, -34), date(2017, 1, 1), date(2017, 7, 1))

    def test_agrupa_csv_y_reutiliza_cache(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bbox = (145., -38., 150., -34.)
            inicio, fin = date(2018, 1, 1), date(2018, 1, 12)
            calls = []
            def simular(b, a, z):
                calls.append((a, z))
                clave = __import__('hashlib').sha256(f"{b}|{a}|{z}|historico".encode()).hexdigest()[:12]
                base = root / 'data' / 'queries' / clave
                p = base / 'processed' / 'modis_filtrado.csv'
                p.parent.mkdir(parents=True, exist_ok=True)
                with p.open('w', encoding='utf-8', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=hb.COLUMNAS)
                    w.writeheader()
                    w.writerow({'latitude': -35, 'longitude': 149, 'acq_date': a.isoformat(), 'acq_time': '1200', 'source': 'MODIS_SP'})
                r = {'bbox': list(b), 'inicio': a.isoformat(), 'fin': z.isoformat(), 'archivos': {'modis': str(p)},
                     'resumen': {'modis': {'recibidos': 1, 'limpios': 1, 'descartados': 0, 'sin_tipo': 0}},
                     'estado_sensores': {'modis': 'descarga_exitosa'}}
                (base / 'consulta.json').write_text(json.dumps(r), encoding='utf-8')
                return r
            fake = types.ModuleType('ingest_historico')
            fake.ejecutar = simular
            with patch.object(hb, 'BASE_DIR', root), patch.dict(sys.modules, {'ingest_historico': fake}):
                r = hb.descargar_historico_por_bloques(bbox, inicio, fin)
                self.assertEqual(len(calls), 3)
                self.assertEqual(r['bloques_completados'], 3)
                with Path(r['archivos']['modis']).open(encoding='utf-8', newline='') as fh:
                    self.assertEqual(len(list(csv.DictReader(fh))), 3)
                hb.descargar_historico_por_bloques(bbox, inicio, fin)
                self.assertEqual(len(calls), 3, 'Debe servir la caché sin volver a descargar')


if __name__ == '__main__':
    unittest.main()
