"""Prueba local de conexión entre Sprint 1, Sprint 2 y Sprint 3 sin llamar NASA."""
import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pipeline_adapter


class IntegracionTests(unittest.TestCase):
    def test_pipeline_completo_con_dos_sensores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = root / 'processed'
            processed.mkdir()
            base = {
                'acq_date': '2019-02-14', 'acq_time': 319,
                'satellite': 'N', 'instrument': 'VIIRS', 'type': 0, 'frp': 12,
                'daynight': 'D', 'year': 2019,
                'timestamp': '2019-02-14T03:19:00Z',
            }
            viirs = dict(base, latitude=-30.0, longitude=148.0, source='viirs-snpp', confidence='h')
            modis = dict(base, latitude=-30.0001, longitude=148.0001, source='modis', confidence='85')
            pd.DataFrame([viirs]).to_csv(processed / 'viirs_snpp_filtrado.csv', index=False)
            pd.DataFrame([modis]).to_csv(processed / 'modis_filtrado.csv', index=False)
            response = {'success': True, 'archivos': {
                'viirs_snpp': str(processed / 'viirs_snpp_filtrado.csv'),
                'modis': str(processed / 'modis_filtrado.csv')}, 'estado_sensores': {'modis': 'descarga_exitosa'}}
            import datetime
            with patch.object(pipeline_adapter, 'request_ingest_preview', return_value=response), \
                 patch.object(pipeline_adapter, 'DATA_DIR', root / 'data'):
                result = pipeline_adapter.request_full_pipeline((147.0, -31.0, 149.0, -29.0),
                                    datetime.date(2019, 2, 14), datetime.date(2019, 2, 15))
            self.assertTrue(result['success'], result.get('message'))
            self.assertEqual(result['status'], 'pipeline_completed')
            df = pd.read_csv(result['csv_path'])
            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]['source'], 'viirs-snpp')
            self.assertEqual(result['analytics_summary']['selected_detections'], 1)
            self.assertEqual(result['analytics_summary']['anomalies_status'], 'sin_referencia_suficiente')
            self.assertTrue((Path(result['analytics_dir']) / 'conteos_mensuales.csv').exists())

    def test_sin_registros(self):
        import datetime
        with patch.object(pipeline_adapter, 'request_ingest_preview', return_value={
                'success': True, 'archivos': {}, 'estado_sensores': {}}):
            res = pipeline_adapter.request_full_pipeline((147, -31, 149, -29),
                                    datetime.date(2019, 2, 14), datetime.date(2019, 2, 15))
        self.assertEqual(res['status'], 'no_detections')

    def test_error_en_descarga(self):
        import datetime
        with patch.object(pipeline_adapter, 'request_ingest_preview', return_value={
                'success': False, 'message': 'Rango demasiado grande'}):
            res = pipeline_adapter.request_full_pipeline((147, -31, 149, -29),
                                    datetime.date(2019, 2, 14), datetime.date(2019, 2, 15))
        self.assertFalse(res['success'])
        self.assertIn('Rango',res['message'])

if __name__ == '__main__':
    unittest.main()
