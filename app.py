"""
Punto de entrada raíz para despliegue en Streamlit Community Cloud.
Redirige la ejecución a src/app.py garantizando compatibilidad tanto si el
usuario especifica 'app.py' como 'src/app.py' en el panel de despliegue.
"""

from pathlib import Path
import sys
import runpy

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

app_path = SRC_DIR / "app.py"
runpy.run_path(str(app_path), run_name="__main__")
