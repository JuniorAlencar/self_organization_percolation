import json
import gzip
import sys
import tempfile
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "process_dynamic_growth.py"
SPEC = spec_from_file_location("process_dynamic_growth", MODULE_PATH)
PROCESS_DYNAMIC_GROWTH = module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = PROCESS_DYNAMIC_GROWTH
SPEC.loader.exec_module(PROCESS_DYNAMIC_GROWTH)

TIME_SERIES_PATH = REPO_ROOT / "jupyter" / "src" / "TimeSeriesAnalysis.py"
TIME_SERIES_SPEC = spec_from_file_location("TimeSeriesAnalysis", TIME_SERIES_PATH)
TIME_SERIES_ANALYSIS = module_from_spec(TIME_SERIES_SPEC)
assert TIME_SERIES_SPEC.loader is not None
sys.modules[TIME_SERIES_SPEC.name] = TIME_SERIES_ANALYSIS
TIME_SERIES_SPEC.loader.exec_module(TIME_SERIES_ANALYSIS)


class ProcessDynamicGrowthTest(unittest.TestCase):
    def _make_data_dir(self, root: Path) -> tuple[Path, Path, Path, Path]:
        raw_root = root / "SOP_data" / "raw_growth_test_dynamic"
        published_root = root / "SOP_data" / "published_dynamic"
        manifests_root = root / "SOP_data" / "manifests_dynamic"
        data_dir = (
            raw_root
            / "S1_percolation"
            / "num_colors_2"
            / "dim_2"
            / "L_8"
            / "fT_constant"
            / "fT_0.3"
            / "c_0.2"
            / "rho_0.2"
            / "data"
        )
        data_dir.mkdir(parents=True, exist_ok=True)
        return raw_root, published_root, manifests_root, data_dir

    def _write_sample(self, data_dir: Path, name: str, pt: list[float]) -> Path:
        sample_path = data_dir / name
        sample_path.write_text(
            json.dumps(
                {
                    "meta": {
                        "t_eq_by_species": [5.0],
                        "z_max": [1.0],
                        "z_stat": [1.0],
                    },
                    "results": {
                        "order_percolation 1": {
                            "data": {
                                "color": 1,
                                "t_eq_species": 5.0,
                                "time": [6.0, 7.0, 8.0],
                                "pt": pt,
                                "nt": [0.1, 0.2, 0.3],
                            }
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return sample_path

    def test_rebuilds_published_bundle_when_new_samples_arrive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root, published_root, manifests_root, data_dir = self._make_data_dir(root)

            self._write_sample(data_dir, "sample_P0_0.7_p0_0.2.json", [0.2, 0.4, 0.6])

            out_path, all_rows, all_color_rows = PROCESS_DYNAMIC_GROWTH.process_group(
                data_dir,
                raw_root,
                published_root,
                manifests_root,
                jobs=2,
            )

            self.assertTrue(out_path.exists())
            self.assertEqual(len(all_rows), 1)
            self.assertEqual(all_rows[0]["N_samples"], 1)
            self.assertEqual(len(all_color_rows), 1)
            self.assertEqual(all_color_rows[0]["N_samples"], 1)

            first_sample_path = data_dir / "sample_P0_0.7_p0_0.2.json"
            first_sample_path.unlink()
            self._write_sample(data_dir, "sample_P0_0.7_p0_0.2_2.json", [0.2, 0.4, 0.6])

            out_path, all_rows, all_color_rows = PROCESS_DYNAMIC_GROWTH.process_group(
                data_dir,
                raw_root,
                published_root,
                manifests_root,
                jobs=2,
            )

            self.assertTrue(out_path.exists())
            self.assertEqual(len(all_rows), 1)
            self.assertEqual(all_rows[0]["N_samples"], 2)
            self.assertEqual(all_color_rows[0]["N_samples"], 2)

            bundle_path = out_path
            with bundle_path.open("r", encoding="utf-8") as handle:
                bundle = json.load(handle)

            self.assertEqual(bundle["p0_groups"][0]["num_samples_total"], 2)
            self.assertEqual(bundle["p0_groups"][0]["orders"][0]["N_samples"], 2)

    def test_updates_time_series_from_published_when_raw_is_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root, published_root, manifests_root, data_dir = self._make_data_dir(root)
            sample_name = "sample_P0_0.7_p0_0.2.json"

            self._write_sample(data_dir, sample_name, [0.2, 0.4, 0.6])
            out_path, _, _ = PROCESS_DYNAMIC_GROWTH.process_group(
                data_dir,
                raw_root,
                published_root,
                manifests_root,
                jobs=2,
            )

            (data_dir / sample_name).unlink()
            self._write_sample(data_dir, sample_name, [0.6, 0.8, 1.0])
            out_path, all_rows, _ = PROCESS_DYNAMIC_GROWTH.process_group(
                data_dir,
                raw_root,
                published_root,
                manifests_root,
                jobs=2,
            )

            self.assertEqual(all_rows[0]["N_samples"], 2)
            with out_path.open("r", encoding="utf-8") as handle:
                bundle = json.load(handle)
            order_data = bundle["p0_groups"][0]["orders"][0]["data"]
            self.assertEqual(order_data["n_seeds_pt"], 2)
            self.assertEqual(order_data["pt_mean"], [0.4, 0.6000000000000001, 0.8])

    def test_process_sample_files_uses_multiple_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, _, _, data_dir = self._make_data_dir(root)
            sample_a = self._write_sample(data_dir, "sample_a_P0_0.7_p0_0.2.json", [0.2, 0.4, 0.6])
            sample_b = self._write_sample(data_dir, "sample_b_P0_0.7_p0_0.2.json", [0.6, 0.8, 1.0])

            rows, stabilized_counts = PROCESS_DYNAMIC_GROWTH.process_sample_files(
                [sample_a, sample_b],
                jobs=2,
            )

            self.assertEqual(stabilized_counts, [1.0, 1.0])
            self.assertEqual(len(rows), 2)

    def test_process_lateral_compact_correlation_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = root / "sample_lateral_correlation_time.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "sample_id,dim,L,t,f_t,r_max,n_rows,C_norm_mean,C_norm_std,C_norm_absmax,r_at_absmax,valid_norm_mean,pair_count_mean,boundary_mode",
                        "sample,3,8,10,0.2,4,5,0.1,0.02,0.5,2,1,128,periodic",
                        "sample,3,8,11,0.3,4,5,0.2,0.03,0.6,3,1,128,periodic",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            rows, counts = PROCESS_DYNAMIC_GROWTH.process_correlation_files([csv_path])

            self.assertEqual(counts, [2.0])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["series_kind"], "correlation_summary")
            self.assertEqual(rows[0]["series"][0]["C_norm_mean"], 0.1)
            self.assertEqual(rows[0]["series"][1]["r_at_absmax"], 3.0)

    def test_lateral_bundle_is_written_compressed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, _, _, data_dir = self._make_data_dir(root)
            correlations_dir = data_dir.parent / "correlations"
            correlations_dir.mkdir(parents=True, exist_ok=True)
            (correlations_dir / "sample_lateral_correlation_time.csv").write_text(
                "\n".join(
                    [
                        "sample_id,dim,L,t,f_t,r_max,n_rows,C_norm_mean,C_norm_std,C_norm_absmax,r_at_absmax,valid_norm_mean,pair_count_mean,boundary_mode",
                        "sample,3,8,10,0.2,4,5,0.1,0.02,0.5,2,1,128,periodic",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            out_dir = root / "published"
            out_dir.mkdir()
            out_path = PROCESS_DYNAMIC_GROWTH.process_lateral_correlations(data_dir, out_dir)

            self.assertEqual(out_path.name, "lateral_correlations_bundle.json.gz")
            self.assertTrue(out_path.exists())
            with gzip.open(out_path, "rt", encoding="utf-8") as handle:
                bundle = json.load(handle)
            self.assertEqual(bundle["meta"]["format"], "compact_summary")

            loaded_by_file = TIME_SERIES_ANALYSIS.load_lateral_correlations_bundle(out_path)
            loaded_by_dir = TIME_SERIES_ANALYSIS.load_lateral_correlations_bundle(out_dir)
            self.assertEqual(loaded_by_file["meta"]["format"], "compact_summary")
            self.assertEqual(loaded_by_dir["meta"]["format"], "compact_summary")

            rows = TIME_SERIES_ANALYSIS.iter_lateral_series_rows(out_path, obs_type="correlation")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["sample_id"], "sample")
            self.assertEqual(rows[0]["C_norm_mean"], 0.1)


if __name__ == "__main__":
    unittest.main()
