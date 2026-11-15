import sys
import unittest
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from period_anomalies import calcular_variaciones_periodo


class PeriodVariationsTests(unittest.TestCase):
    def test_ventana_93_dias(self):
        datos = pd.DataFrame({"timestamp": pd.date_range("2019-01-01", periods=93, freq="D")})
        df, med = calcular_variaciones_periodo(datos, "2019-01-01", "2019-04-03", "Semanal")
        self.assertEqual(df["dias"].sum(), 93)
        self.assertEqual(df["detecciones"].sum(), 93)
        self.assertAlmostEqual(med, 1.0)

    def test_ceros_dentro_del_intervalo(self):
        datos = pd.DataFrame({"timestamp": ["2019-01-02", "2019-01-02"]})
        df, med = calcular_variaciones_periodo(datos, "2019-01-01", "2019-01-07", "Diaria")
        self.assertEqual(len(df), 7)
        self.assertEqual(df["detecciones"].sum(), 2)
        self.assertEqual(med, 0.0)
        self.assertTrue(df["variacion_pct"].isna().all())

    def test_semanas_parciales_normalizadas(self):
        fechas = ["2019-01-01"] * 7 + ["2019-01-09"] * 2
        df, med = calcular_variaciones_periodo(pd.DataFrame({"timestamp": fechas}), "2019-01-01", "2019-01-09", "Semanal")
        self.assertEqual(df["dias"].tolist(), [7, 2])
        self.assertEqual(df["tasa_diaria"].tolist(), [1, 1])

    def test_meses_parciales(self):
        df, med = calcular_variaciones_periodo(pd.DataFrame({"timestamp": ["2019-01-31", "2019-02-01"]}), "2019-01-31", "2019-02-01", "Mensual")
        self.assertEqual(df["dias"].tolist(), [1, 1])

    def test_incluye_limites(self):
        df, med = calcular_variaciones_periodo(pd.DataFrame({"timestamp": ["2019-01-01", "2019-01-04"]}), "2019-01-01", "2019-01-04", "Diaria")
        self.assertEqual(df["detecciones"].sum(), 2)

    def test_fechas_invertidas(self):
        with self.assertRaises(ValueError):
            calcular_variaciones_periodo(pd.DataFrame({"timestamp": []}), "2020-01-02", "2020-01-01")

    def test_no_mutar_dataframe(self):
        datos = pd.DataFrame({"timestamp": ["2019-01-01"], "source": ["MODIS"]})
        original = datos.copy(deep=True)
        calcular_variaciones_periodo(datos, "2019-01-01", "2019-01-07")
        pd.testing.assert_frame_equal(datos, original)

    def test_fuera_periodo(self):
        df, med = calcular_variaciones_periodo(pd.DataFrame({"timestamp": ["2017-01-01", "2019-01-03"]}), "2019-01-01", "2019-01-07", "Diaria")
        self.assertEqual(df["detecciones"].sum(), 1)


if __name__ == "__main__":
    unittest.main()
