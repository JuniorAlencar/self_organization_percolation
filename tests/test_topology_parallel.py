import csv
import json
import lzma
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import process_counts as counts
import process_raw_fractions as fractions
from topology_parallel import ordered_parallel_map
from test_process_spanning_counts import cluster


def fail_worker(value):
    raise ValueError('worker failure')


class TopologyParallelTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw = self.root / ('raw_fractions/bond_percolation/num_colors_1/dim_2/L_32/'
                                'fT_constant/fT_0.1/c_0.1/rho_1.0/gap_1.0L')
        (self.raw / 'counts').mkdir(parents=True)
        self.payload = {'meta': {'schema_version': 2}, 'samples': [
            {'sample_index': 0, 'num_spanning_clusters': 2,
             'spanning_clusters': [cluster(0, 64), cluster(1, 32)]},
            {'sample_index': 1, 'num_spanning_clusters': 0, 'spanning_clusters': []}]}
        for seed in (44, 45, 46):
            (self.raw / 'counts' / f'light_seed_{seed}_ts_test_P0_1.00_p0_1.00_counts.topology').write_text(json.dumps(self.payload))

    def args(self, out='parallel', jobs=2):
        return counts.build_parser().parse_args(['--sop-root', str(self.root), '--out-dir', out, '--jobs', str(jobs)])

    def test_serial_and_parallel_counts_identical(self):
        counts.process_counts(self.args('serial', 1))
        counts.process_counts(self.args('parallel', 2))
        for first in (self.root / 'serial').glob('**/counts_P0_*'):
            second = self.root / 'parallel' / first.relative_to(self.root / 'serial')
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_cache_skip_and_invalidation(self):
        args = self.args(jobs=1)
        counts.process_counts(args)
        with patch.object(counts, 'process_count_file', side_effect=AssertionError('parsed raw')):
            self.assertTrue(counts.process_counts(args)['skipped_unchanged'])
        # Output damage and input changes both invalidate the shortcut.
        output = next((self.root / 'parallel').glob('**/*.clusters.csv'))
        output.unlink()
        self.assertNotIn('skipped_unchanged', counts.process_counts(args))
        self.assertTrue(output.exists())
        path = next((self.raw / 'counts').glob('*.topology'))
        path.write_text(json.dumps(self.payload) + '\n')
        self.assertNotIn('skipped_unchanged', counts.process_counts(args))
        args.box_epsilon_min = 4
        self.assertNotIn('skipped_unchanged', counts.process_counts(args))
        args.refresh = True
        self.assertNotIn('skipped_unchanged', counts.process_counts(args))
        args.refresh = False
        (self.root / 'parallel' / '.counts_state.json').write_text('[]')
        self.assertNotIn('skipped_unchanged', counts.process_counts(args))

    def test_new_and_removed_input_invalidate_cache(self):
        args = self.args(jobs=1)
        counts.process_counts(args)
        new = self.raw / 'counts/light_seed_47_ts_test_P0_1.00_p0_1.00_counts.topology'
        new.write_text(json.dumps(self.payload))
        self.assertNotIn('skipped_unchanged', counts.process_counts(args))
        new.unlink()
        self.assertNotIn('skipped_unchanged', counts.process_counts(args))

    def test_parallel_fraction_bundles_match_serial_and_skip(self):
        for P0 in ('0.20', '0.30'):
            for seed in (44, 45):
                payload = {'meta': {'dim': 2, 'L': 32, 'seed': seed},
                           'data': {'S_stab': [64, 32], 'anchor_z': [0, 32]}}
                (self.raw / f'light_seed_{seed}_ts_test_P0_{P0}_p0_1.00.json').write_text(json.dumps(payload))
        for jobs, label in [(1, 'serial'), (2, 'parallel')]:
            paths = fractions.process_raw_fractions(self.root, published_dir=label,
                     manifests_dir=label+'_manifests', jobs=jobs)
            self.assertEqual(len(paths), 2)
        for path in (self.root / 'serial').rglob('*.xz'):
            other = self.root / 'parallel' / path.relative_to(self.root / 'serial')
            with lzma.open(path, 'rt') as f: first = json.load(f)
            with lzma.open(other, 'rt') as f: second = json.load(f)
            first['meta'].pop('processed_at_utc')
            second['meta'].pop('processed_at_utc')
            self.assertEqual(first, second)
        before = {p: p.stat().st_mtime_ns for p in (self.root / 'parallel').rglob('*.xz')}
        fractions.process_raw_fractions(self.root, published_dir='parallel',
                 manifests_dir='parallel_manifests', jobs=2)
        self.assertEqual(before, {p: p.stat().st_mtime_ns for p in before})

    def test_worker_errors_propagate_and_jobs_validate(self):
        with self.assertRaisesRegex(ValueError, 'worker failure'):
            list(ordered_parallel_map(fail_worker, [1, 2], 2))
        script = Path(counts.__file__).with_name('update_topological.sh')
        result = subprocess.run(['bash', str(script), '--jobs', '0'], capture_output=True)
        self.assertEqual(result.returncode, 2)

    def test_parallel_shell_pipeline(self):
        script = Path(counts.__file__).with_name('update_topological.sh')
        result = subprocess.run(['bash', str(script), '--counts-only', '--jobs', '2',
                                 '--sop-root', str(self.root)], check=True, capture_output=True, text=True)
        self.assertIn('jobs=2', result.stdout)
        paths = list((self.root / 'published_counts').glob('**/*.samples.csv'))
        with paths[0].open() as f: self.assertEqual(len(list(csv.DictReader(f))), 6)


if __name__ == '__main__':
    unittest.main()
