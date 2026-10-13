import os
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/counts_analysis_mpl')
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jupyter'))
from src import CountsAnalysis as analysis


def cluster(index):
    return {'cluster_index': index, 'cluster_role': 'largest_spanning' if index == 0 else 'other_spanning',
            'properties': {'d_bulk': {'epsilon': [2,4,8], 'L_over_epsilon': [16,8,4], 'N_bulk': [256,64,16]},
                           'd_min': {'R': [2,4,8], 'L_over_R': [16,8,4], 'N_R': [16,8,4]},
                           'd_min_chemical': {'r': [1,2,3], 'ell_min': [1,100,10000]}}}


class CountsAnalysisSpanningTest(unittest.TestCase):
    def setUp(self):
        self.raw = {'meta': {}, 'samples': [
            {'sample_id': 'a', 'sample_index': 0, 'spanning_clusters': [cluster(0), cluster(1), cluster(2)]},
            {'sample_id': 'b', 'sample_index': 1, 'spanning_clusters': []},
            {'sample_id': 'legacy', 'properties': {'d_min': {'ell_min': [1,2]}}},
        ]}

    def test_roles_and_yardstick_only(self):
        largest = analysis.select_spanning_clusters(self.raw)
        self.assertEqual(len(largest['samples']), 1)
        fits = analysis.fit_all_observables(largest)
        self.assertEqual(set(fits['observable']), {'d_f','d_min'})
        self.assertAlmostEqual(fits.loc[fits.observable == 'd_f', 'slope'].iloc[0], 2)
        self.assertAlmostEqual(fits.loc[fits.observable == 'd_min', 'slope'].iloc[0], 1)
        others = analysis.select_spanning_clusters(self.raw, 'other_spanning')
        self.assertEqual([s['cluster_index'] for s in others['samples']], [1,2])
        mean = analysis.aggregate_series(others, 'd_min')
        self.assertEqual(mean.n_clusters.tolist(), [2,2,2])
        self.assertEqual(mean.n_samples.tolist(), [1,1,1])
        self.assertEqual(set(analysis.summarize_fits(analysis.fit_all_observables(others)).cluster_role), {'other_spanning'})

    def test_missing_yardstick_is_not_replaced_by_chemical_distance(self):
        self.raw['samples'][0]['spanning_clusters'][0]['properties']['d_min'] = {'r': [1,2], 'ell_min': [1,2]}
        with self.assertRaisesRegex(ValueError, 'yardstick'):
            analysis.select_spanning_clusters(self.raw)

    def test_empty_selection(self):
        empty = analysis.select_spanning_clusters({'samples': []})
        self.assertTrue(analysis.fit_all_observables(empty).empty)
        self.assertTrue(analysis.aggregate_series(empty, 'd_min').empty)
        self.assertIn('y_mean', analysis.aggregate_series(empty, 'd_min').columns)
