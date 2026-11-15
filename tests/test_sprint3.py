import sys
import unittest
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analytics import (filter_region_dates, monthly_yearly_counts, historical_monthly_median,
                       monthly_anomalies, rankings)


def sample():
    return pd.DataFrame([
        {"latitude": -30.0, "longitude": 148.0, "timestamp": "2017-02-14T10:00:00Z", "source": "modis"},
        {"latitude": -30.1, "longitude": 148.1, "timestamp": "2018-02-14T11:00:00Z", "source": "viirs-snpp"},
        {"latitude": -30.2, "longitude": 148.2, "timestamp": "2019-02-14T12:00:00Z", "source": "viirs-snpp"},
        {"latitude": -30.3, "longitude": 148.3, "timestamp": "2020-02-14T13:00:00Z", "source": "viirs-snpp"},
        {"latitude": -30.4, "longitude": 148.4, "timestamp": "2021-02-14T14:00:00Z", "source": "modis"},
        {"latitude": -30.5, "longitude": 148.5, "timestamp": "2022-02-15T01:00:00Z", "source": "viirs-snpp"},
        {"latitude": -45.0, "longitude": 170.0, "timestamp": "2022-02-15T01:00:00Z", "source": "modis"},
    ])


class Sprint3Tests(unittest.TestCase):
    def test_bbox_y_fechas_inclusivas(self):
        out = filter_region_dates(sample(), (147, -32, 150, -28), "2022-02-15", "2022-02-15")
        self.assertEqual(len(out), 1)

    def test_bbox_invalido(self):
        with self.assertRaises(ValueError):
            filter_region_dates(sample(), (150, -32, 147, -28))

    def test_fechas_invertidas(self):
        with self.assertRaises(ValueError):
            filter_region_dates(sample(), start="2022-02-20", end="2022-02-01")

    def test_agrupaciones(self):
        monthly, yearly = monthly_yearly_counts(sample())
        self.assertEqual(monthly.detections.sum(), 7)
        self.assertEqual(yearly.detections.sum(), 7)

    def test_mediana_multianual(self):
        median = historical_monthly_median(sample(), [2017, 2018, 2019, 2020, 2021], full_coverage=True)
        self.assertEqual(float(median.loc[median.month == 2, "historical_median"].iloc[0]), 1)
        self.assertEqual(float(median.loc[median.month == 3, "historical_median"].iloc[0]), 0)

    def test_mediana_sin_cobertura_rechazada(self):
        with self.assertRaises(ValueError):
            historical_monthly_median(sample(), [2017, 2018, 2019, 2020, 2021])

    def test_mediana_pocos_anios_rechazada(self):
        with self.assertRaises(ValueError):
            historical_monthly_median(sample(), [2018, 2019], full_coverage=True)

    def test_anomalias(self):
        res = monthly_anomalies(sample(), [2017, 2018, 2019, 2020, 2021], full_coverage=True)
        row = res.loc[(res.year == 2022) & (res.month == 2)].iloc[0]
        self.assertEqual(row.detections, 2)
        self.assertEqual(row.anomaly_absolute, 1)
        self.assertEqual(row.anomaly_percent, 100)

    def test_rankings(self):
        ys, ms = rankings(sample())
        self.assertEqual(int(ys.iloc[0].year), 2022)
        self.assertEqual(int(ms.iloc[0].detections), 2)


if __name__ == "__main__":
    unittest.main()
