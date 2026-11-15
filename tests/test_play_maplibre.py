"""Pruebas del reproductor con el API MapLibre de Plotly (versiones 5.24+)."""
import datetime as dt
import sys
import unittest
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from temporal_play import build_timeline_figure

class PlayMapLibreTests(unittest.TestCase):
    def test_reproductor_mapa_moderno_y_controles(self):
        data = pd.DataFrame([
            {"timestamp":"2019-02-24T03:00:00Z","latitude":-29.0,"longitude":145.0,"source":"MODIS","frp":8.0},
            {"timestamp":"2019-02-25T03:00:00Z","latitude":-29.1,"longitude":145.1,"source":"VIIRS","frp":15.0},
        ])
        fig, info = build_timeline_figure(data,(140.0,-34.0,150.0,-26.0),dt.date(2019,2,24),dt.date(2019,2,25),frequency="Diaria")
        self.assertIsInstance(fig.data[0], go.Scattermap)
        self.assertEqual(fig.frames[0].data[0].type,"scattermap")
        self.assertIsNotNone(fig.layout.map)
        self.assertEqual(fig.layout.map.style,"open-street-map")
        self.assertEqual(info["total_records"],2)
        self.assertEqual(info["frames"],2)
        self.assertIn("Play",fig.layout.updatemenus[0].buttons[0].label)
    def test_periodo_sin_detecciones(self):
        df=pd.DataFrame(columns=["timestamp","latitude","longitude"])
        fig, info=build_timeline_figure(df,(140.,-34.,150.,-26.),dt.date(2019,2,24),dt.date(2019,2,25),frequency="Diaria")
        self.assertEqual(info["total_records"],0)
        self.assertEqual(len(fig.frames),2)
        self.assertEqual(fig.frames[0].data[0].type,"scattermap")

if __name__=="__main__": unittest.main()
