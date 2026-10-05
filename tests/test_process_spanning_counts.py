import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import process_counts as counts


def cluster(index, mass):
    return {
        "cluster_index": index,
        "component": {"num_sites": mass, "num_bonds": mass - 1, "component_seed": index},
        "d_bulk": {"value": 1.8, "r_squared": 0.99, "num_scales": 3},
        "d_min": {"value": None, "reason_if_undefined": "fewer_than_three_valid_scales"},
        "box_counting_component": {"counts": [
            {"epsilon": 2, "offset_id": 0, "num_boxes": 10},
            {"epsilon": 2, "offset_id": 1, "num_boxes": 12},
            {"epsilon": 4, "offset_id": 0, "num_boxes": 4},
        ]},
        "minimum_path_yardstick": {"defined": True, "path_length": 31,
                                   "counts": [{"R": 2, "num_spheres": 16}]},
    }


class SpanningCountsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / (
            "raw_fractions/bond_percolation/num_colors_1/dim_2/L_32/"
            "fT_constant/fT_0.1/c_0.1/rho_1.0/gap_1.0L/counts/"
            "light_seed_44_ts_test_P0_1.00_p0_1.00_counts.topology"
        )
        self.path.parent.mkdir(parents=True)
        self.payload = {"meta": {"schema_version": 2}, "samples": [
            {"sample_index": 0, "anchor_z": 7, "t_stab": 100,
             "num_spanning_clusters": 2, "spanning_clusters": [cluster(1, 32), cluster(0, 64)]},
            {"sample_index": 1, "anchor_z": 40, "t_stab": 200,
             "num_spanning_clusters": 0, "spanning_clusters": []},
        ]}
        self.write_raw()
        self.args = counts.build_parser().parse_args(["--sop-root", str(self.root)])

    def write_raw(self):
        self.path.write_text(json.dumps(self.payload))

    def output(self):
        path = next((self.root / "published_counts").glob("**/counts_P0_*.json"))
        return path, json.loads(path.read_text())

    def test_roles_curves_fits_and_zero_sample_csv(self):
        counts.process_counts(self.args)
        path, data = self.output()
        first, empty = data["samples"]
        self.assertEqual(empty["num_spanning_clusters"], 0)
        self.assertIsNone(empty["largest_spanning_cluster_index"])
        big, other = first["spanning_clusters"]
        self.assertEqual(big["cluster_role"], "largest_spanning")
        self.assertEqual(other["cluster_role"], "other_spanning")
        self.assertEqual(big["component"]["num_sites"], 64)
        self.assertEqual(big["properties"]["d_bulk"]["N_bulk"], [11, 4])
        self.assertEqual(big["properties"]["d_min"]["N_R"], [16])
        self.assertEqual(big["d_bulk"]["value"], 1.8)
        self.assertIsNone(big["d_min"]["value"])
        self.assertNotIn("d_hull", big["properties"])
        self.assertEqual(data["meta"]["n_spanning_clusters"], 2)
        self.assertEqual(data["meta"]["n_samples_without_spanning"], 1)
        self.assertEqual(data["meta"]["n_d_min_yardstick_samples"], 1)
        with path.with_suffix('.clusters.csv').open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([r["cluster_role"] for r in rows], ["largest_spanning", "other_spanning"])
        self.assertEqual(rows[0]["d_min_value"], "")
        with path.with_suffix('.samples.csv').open() as f:
            self.assertEqual([r['num_spanning_clusters'] for r in csv.DictReader(f)], ['2', '0'])

    def test_reprocessing_replaces_cluster_list_without_duplicates(self):
        counts.process_counts(self.args)
        counts.process_counts(self.args)
        self.assertEqual(len(self.output()[1]["samples"]), 2)
        self.payload["samples"][0]["spanning_clusters"] = [cluster(0, 10)]
        self.payload["samples"][0]["num_spanning_clusters"] = 1
        self.write_raw()
        counts.process_counts(self.args)
        path, data = self.output()
        self.assertEqual(data["meta"]["n_spanning_clusters"], 1)
        with path.with_suffix('.clusters.csv').open() as f:
            self.assertEqual(len(list(csv.DictReader(f))), 1)
        self.path.unlink()
        counts.process_counts(self.args)
        self.assertEqual(self.output()[1]["samples"], data["samples"])

    def test_legacy_not_misclassified_as_spanning(self):
        legacy = cluster(0, 64)
        legacy.update(sample_index=2, eligible=True)
        self.payload["samples"].append(legacy)
        self.write_raw()
        counts.process_counts(self.args)
        counts.process_counts(self.args)
        data = self.output()[1]
        old = data["samples"][2]
        self.assertEqual(old["cluster_role"], "legacy_largest_component_spanning_unknown")
        self.assertIsNone(old["num_spanning_clusters"])
        self.assertNotIn("spanning_clusters", old)
        self.assertEqual(data["meta"]["n_legacy_samples"], 1)

    def test_dry_run_leaves_existing_outputs_unchanged(self):
        counts.process_counts(self.args)
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.args.dry_run = True
        counts.process_counts(self.args)
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_invalid_count_rejected(self):
        self.payload["samples"][0]["num_spanning_clusters"] = 3
        self.write_raw()
        with self.assertRaisesRegex(ValueError, "num_spanning_clusters"):
            counts.process_counts(self.args)

    def test_shell_counts_only_entrypoint(self):
        result = subprocess.run(['bash', str(TOOLS / 'update_topological.sh'),
                                 '--counts-only', '--sop-root', str(self.root)],
                                check=True, capture_output=True, text=True)
        self.assertIn('process_counts', result.stdout)
        self.assertEqual(self.output()[1]['meta']['n_spanning_clusters'], 2)
        self.assertFalse((self.root / 'published_dynamic').exists())


if __name__ == '__main__':
    unittest.main()
