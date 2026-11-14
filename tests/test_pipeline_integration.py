"""
Pruebas automatizadas de la integración del Pipeline (Sprints 1 → 2 → 3 → 4).
NASA Space Apps Challenge 2026 - Harmonization of MODIS and VIIRS Hot Spots.
"""

import unittest
import datetime
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline_adapter import (
    check_pipeline_modules,
    is_bbox_covered_by_colombia,
    get_query_hash,
    get_required_contracts_doc,
    request_pipeline_execution,
    BBOX_COLOMBIA_COBERTURA,
)


class TestPipelineIntegration(unittest.TestCase):

    def test_check_pipeline_modules_detects_empty_files(self):
        status = check_pipeline_modules()
        self.assertIn("config", status)
        self.assertIn("ingest", status)
        self.assertIn("harmonize", status)
        self.assertIn("analytics", status)

        # Verificar que reporta adecuadamente el estado actual
        for mod in ["ingest", "harmonize", "analytics"]:
            self.assertIn(status[mod]["status"], ["empty", "missing", "present"])
            self.assertIsInstance(status[mod]["has_required_interface"], bool)

    def test_is_bbox_covered_by_colombia(self):
        # Punto céntrico de Colombia (Bogotá)
        self.assertTrue(is_bbox_covered_by_colombia((-75.0, 4.0, -73.0, 5.0)))
        # Cobertura completa Colombia
        self.assertTrue(is_bbox_covered_by_colombia(BBOX_COLOMBIA_COBERTURA))
        # Región externa (Amazonas brasileño)
        self.assertFalse(is_bbox_covered_by_colombia((-60.0, -10.0, -50.0, -2.0)))
        # Región externa (Europa)
        self.assertFalse(is_bbox_covered_by_colombia((2.0, 48.0, 3.0, 49.0)))

    def test_get_query_hash_deterministic(self):
        bbox = (-75.0, 4.0, -74.0, 5.0)
        d1 = datetime.date(2018, 1, 1)
        d2 = datetime.date(2018, 12, 31)
        h1 = get_query_hash(bbox, d1, d2)
        h2 = get_query_hash(bbox, d1, d2)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 12)

    def test_get_required_contracts_doc_completeness(self):
        contracts = get_required_contracts_doc()
        self.assertIn("sprint_1_ingest", contracts)
        self.assertIn("sprint_2_harmonize", contracts)
        self.assertIn("sprint_3_analytics", contracts)

        for sprint_key, doc in contracts.items():
            self.assertIn("modulo", doc)
            self.assertIn("firma_requerida", doc)
            self.assertIn("parametros", doc)
            self.assertIn("salidas_esperadas", doc)
            self.assertIn("regla_cientifica", doc)

    def test_request_pipeline_colombia_success(self):
        # Consulta en región dentro de Colombia y rango temporal válido
        bbox = (-75.0, 4.0, -73.0, 5.0)
        d1 = datetime.date(2018, 1, 1)
        d2 = datetime.date(2019, 12, 31)
        result = request_pipeline_execution(bbox, d1, d2)

        self.assertTrue(result["success"])
        self.assertEqual(result["status"], "cached_dataset_available")
        self.assertTrue(result["can_visualize"])

    def test_request_pipeline_external_region_requires_modules(self):
        # Consulta fuera de Colombia
        bbox_external = (-60.0, -10.0, -50.0, -2.0)
        d1 = datetime.date(2018, 1, 1)
        d2 = datetime.date(2019, 12, 31)
        result = request_pipeline_execution(bbox_external, d1, d2)

        # Como los módulos están vacíos o ausentes, no debe simular éxito falso
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "modules_not_ready")
        self.assertFalse(result["can_visualize"])
        self.assertIn("pending_modules", result)
        self.assertIn("contracts", result)

    def test_request_pipeline_invalid_inputs_handled(self):
        # Coordenadas invertidas
        result = request_pipeline_execution((-70.0, 5.0, -75.0, 4.0), datetime.date(2018, 1, 1), datetime.date(2019, 12, 31))
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "invalid_coordinates")

        # Fechas invertidas
        result = request_pipeline_execution((-75.0, 4.0, -73.0, 5.0), datetime.date(2020, 1, 1), datetime.date(2018, 12, 31))
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "invalid_dates")


if __name__ == "__main__":
    unittest.main()
