import sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from activity_zones import summarize_activity_zones

class ActivityZonesTests(unittest.TestCase):
    def test_counts_preserved(self):
        df = pd.DataFrame({'latitude':[0.1,0.1,0.8], 'longitude':[0.1,0.1,0.9]})
        out = summarize_activity_zones(df, (0,0,1,1), 5)
        self.assertEqual(out['detecciones'].sum(), 3)
        self.assertEqual(out['detecciones'].max(), 2)
        self.assertEqual(len(out), 25)
    def test_bbox_boundary_inclusive(self):
        df = pd.DataFrame({'latitude':[0,1], 'longitude':[0,1]})
        self.assertEqual(summarize_activity_zones(df, (0,0,1,1))['detecciones'].sum(), 2)
    def test_invalid_records_ignored(self):
        df = pd.DataFrame({'latitude':[0.2,99,None], 'longitude':[0.2,0,0]})
        self.assertEqual(summarize_activity_zones(df, (0,0,1,1))['detecciones'].sum(), 1)
    def test_empty(self):
        df = pd.DataFrame({'latitude':[], 'longitude':[]})
        self.assertEqual(summarize_activity_zones(df, (0,0,1,1))['detecciones'].sum(), 0)
    def test_invalid_bbox(self):
        with self.assertRaises(ValueError):
            summarize_activity_zones(pd.DataFrame({'latitude':[1],'longitude':[1]}),(1,1,0,0))
    def test_input_unchanged(self):
        df = pd.DataFrame({'latitude':[0.2], 'longitude':[0.2]})
        before = df.copy(deep=True)
        summarize_activity_zones(df, (0,0,1,1))
        pd.testing.assert_frame_equal(df, before)

if __name__ == '__main__':
    unittest.main()
