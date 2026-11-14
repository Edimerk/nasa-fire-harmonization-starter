"""
Prueba automatizada de preparación para despliegue (Streamlit Community Cloud).
NASA Space Apps Challenge 2026 - Harmonization of MODIS and VIIRS Hot Spots.
"""

import unittest
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent

class TestDeploymentReadiness(unittest.TestCase):

    def test_entrypoints_exist(self):
        root_app = BASE_DIR / "app.py"
        src_app = BASE_DIR / "src" / "app.py"
        self.assertTrue(root_app.exists(), "Falta punto de entrada raíz app.py")
        self.assertTrue(src_app.exists(), "Falta src/app.py")

    def test_streamlit_config_secure(self):
        config_path = BASE_DIR / ".streamlit" / "config.toml"
        self.assertTrue(config_path.exists(), "Falta .streamlit/config.toml")
        content = config_path.read_text(encoding="utf-8")
        self.assertIn("headless = true", content)
        self.assertIn("enableCORS = true", content)
        self.assertIn("enableXsrfProtection = true", content)

    def test_requirements_file(self):
        req_path = BASE_DIR / "requirements.txt"
        self.assertTrue(req_path.exists(), "Falta requirements.txt")
        content = req_path.read_text(encoding="utf-8").lower()
        for pkg in ["streamlit", "folium", "streamlit-folium", "shapely", "plotly", "pandas", "numpy"]:
            self.assertIn(pkg, content, f"Falta {pkg} en requirements.txt")

    def test_python_version_file(self):
        py_ver_path = BASE_DIR / ".python-version"
        self.assertTrue(py_ver_path.exists(), "Falta .python-version")
        ver = py_ver_path.read_text(encoding="utf-8").strip()
        self.assertTrue(ver.startswith("3."), f"Versión de Python inválida: {ver}")

    def test_gitignore_protects_credentials_and_heavy_data(self):
        gi_path = BASE_DIR / ".gitignore"
        self.assertTrue(gi_path.exists(), "Falta .gitignore")
        content = gi_path.read_text(encoding="utf-8")
        self.assertIn("secrets.toml", content, "Debe ignorar secrets.toml")
        self.assertIn("data/raw/", content, "Debe ignorar data/raw/ para no saturar el repositorio")
        self.assertIn("data/processed/", content, "Debe ignorar data/processed/")

    def test_harmonized_dataset_available_and_ram_optimized(self):
        data_path = BASE_DIR / "data" / "harmonized" / "incendios_armonizados.csv"
        self.assertTrue(data_path.exists(), "Falta data/harmonized/incendios_armonizados.csv")

        # Cargar con analytics_core / app.py optimizado
        sys.path.insert(0, str(BASE_DIR / "src"))
        from app import load_harmonized_data
        df, err = load_harmonized_data()
        self.assertEqual(err, "", f"Error cargando dataset: {err}")
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 521534)

        # Verificar huella de RAM optimizada (< 50 MB)
        ram_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        self.assertLess(ram_mb, 50.0, f"Uso de RAM muy alto: {ram_mb:.2f} MB (límite 50 MB)")


if __name__ == "__main__":
    unittest.main()
