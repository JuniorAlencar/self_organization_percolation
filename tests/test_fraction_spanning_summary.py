import json
import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import process_raw_fractions as fractions
from test_process_spanning_counts import cluster


class FractionSpanningTest(unittest.TestCase):
    def test_summary_distinguishes_multiple_zero_legacy_missing_and_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'run.json'
            (path.parent / 'counts').mkdir()
            top = path.parent / 'counts/run_counts.topology'
            data = {'anchor_z': [7, 40, 80, 120], 't_stab': [100, 200, 300, 400]}
            payload = {'samples': [
                {'sample_index': 0, 'anchor_z': 7, 'num_spanning_clusters': 2,
                 'spanning_clusters': [cluster(1, 32), cluster(0, 64)]},
                {'sample_index': 1, 'num_spanning_clusters': 0, 'spanning_clusters': []},
                {'sample_index': 2, 'largest_component': {'num_sites': 64}},
            ]}
            top.write_text(json.dumps(payload))
            rows = list(fractions.spanning_summaries(path, data, 4))
            self.assertEqual(rows[0]['largest_spanning_cluster']['component']['num_sites'], 64)
            self.assertEqual(rows[0]['other_spanning_clusters'][0]['cluster_role'], 'other_spanning')
            self.assertEqual(rows[1]['num_spanning_clusters'], 0)
            self.assertEqual(rows[1]['other_spanning_clusters'], [])
            self.assertEqual(rows[2]['topology_status'], 'legacy_spanning_unknown')
            self.assertIsNone(rows[2]['num_spanning_clusters'])
            self.assertEqual(rows[3]['topology_status'], 'missing_topology')
            self.assertIsNone(rows[3]['other_spanning_clusters'])
            old = fractions.topology_fingerprints([path])
            payload['samples'][0]['anchor_z'] = 8
            top.write_text(json.dumps(payload))
            self.assertNotEqual(old, fractions.topology_fingerprints([path]))
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                list(fractions.spanning_summaries(path, data, 4))

    def test_bundle_keeps_array_alignment_across_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'counts').mkdir()
            paths = [root / 'first.json', root / 'second.json']
            for p in paths:
                p.write_text(json.dumps({'meta': {}, 'data': {'anchor_z': [7], 'S_stab': [999]}}))
            (root / 'counts/second_counts.topology').write_text(json.dumps({'samples': [
                {'sample_index': 0, 'anchor_z': 7, 'num_spanning_clusters': 1,
                 'spanning_clusters': [cluster(0, 32)]}]}))
            result = fractions.build_published_fractions_bundle(paths, root, Path('.'), '1', '1', None)
            self.assertEqual(result['data']['S_stab'], [999, 999])
            self.assertEqual(result['data']['num_spanning_clusters'], [None, 1])
            self.assertEqual(result['meta']['n_samples_with_topology'], 1)
            self.assertEqual({len(v) for v in result['data'].values()}, {2})
