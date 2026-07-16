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
                        "growth_test_stop_criterion": "alive_species_pt_derivative_stability_or_death",
                        "growth_test_t_eq_validation": "discrete_derivative_of_blocked_pt_variation",
                        "growth_test_t_eq_s_prime_threshold": 1.0e-5,
                        "growth_test_equilibrium_effective_rel_tol": 2.5e-3,
                        "growth_test_post_equilibrium_extra_steps": 100,
                        "growth_test_equilibrium_rel_tol_scaling": "fixed_base_tol_times_0p10",
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

    def test_new_dynamic_layout_uses_zero_stat_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, _, _, data_dir = self._make_data_dir(root)

            meta = PROCESS_DYNAMIC_GROWTH.parse_data_dir(data_dir)

            self.assertIsNotNone(meta)
            self.assertEqual(meta["stat_window"], 0)

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
            self.assertEqual(
                bundle["meta"]["stop_criterion"],
                "alive_species_pt_derivative_stability_or_death",
            )
            self.assertEqual(
                bundle["meta"]["t_eq_validation"],
                "discrete_derivative_of_blocked_pt_variation",
            )
            self.assertAlmostEqual(bundle["meta"]["t_eq_s_prime_threshold"], 1.0e-5)
            self.assertAlmostEqual(bundle["meta"]["equilibrium_effective_rel_tol"], 2.5e-3)
            self.assertEqual(bundle["meta"]["post_equilibrium_extra_steps"], 100)
            self.assertEqual(all_rows[0]["stat_window"], 0)
            self.assertAlmostEqual(all_rows[0]["t_eq_s_prime_threshold"], 1.0e-5)
            self.assertEqual(all_rows[0]["post_equilibrium_extra_steps"], 100)

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

    def test_merge_uses_new_series_when_existing_series_is_empty(self) -> None:
        existing_order = {
            "order": 0,
            "N_samples": 1,
            "N_samples_perc": 1,
            "data": {
                "time": [],
                "pt_mean": [],
                "pt_std": [],
                "pt_sem": [],
                "ft_mean": [],
                "ft_std": [],
                "ft_sem": [],
                "n_seeds_pt": 0,
                "n_seeds_ft": 0,
            },
            "p": {"mean": 0.2, "err": 0.0, "n": 1, "sum": 0.2, "sumsq": 0.04},
            "f": {"mean": 0.1, "err": 0.0, "n": 1, "sum": 0.1, "sumsq": 0.01},
            "t_eq_species": {"mean": 5.0, "err": 0.0, "n": 1, "sum": 5.0, "sumsq": 25.0},
            "z_max": {"mean": 1.0, "err": 0.0, "n": 1, "sum": 1.0, "sumsq": 1.0},
            "z_stat": {"mean": 1.0, "err": 0.0, "n": 1, "sum": 1.0, "sumsq": 1.0},
            "samples": [],
        }
        new_order = {
            "order": 0,
            "N_samples": 1,
            "N_samples_perc": 1,
            "data": {
                "time": [6.0, 7.0, 8.0],
                "pt_mean": [0.2, 0.4, 0.6],
                "pt_std": [0.0, 0.0, 0.0],
                "pt_sem": [0.0, 0.0, 0.0],
                "ft_mean": [0.1, 0.2, 0.3],
                "ft_std": [0.0, 0.0, 0.0],
                "ft_sem": [0.0, 0.0, 0.0],
                "n_seeds_pt": 1,
                "n_seeds_ft": 1,
            },
            "p": {"mean": 0.4, "err": 0.0, "n": 1, "sum": 0.4, "sumsq": 0.16},
            "f": {"mean": 0.2, "err": 0.0, "n": 1, "sum": 0.2, "sumsq": 0.04},
            "t_eq_species": {"mean": 5.0, "err": 0.0, "n": 1, "sum": 5.0, "sumsq": 25.0},
            "z_max": {"mean": 1.0, "err": 0.0, "n": 1, "sum": 1.0, "sumsq": 1.0},
            "z_stat": {"mean": 1.0, "err": 0.0, "n": 1, "sum": 1.0, "sumsq": 1.0},
            "samples": [],
        }

        merged = PROCESS_DYNAMIC_GROWTH.merge_order_block(existing_order, new_order)

        self.assertEqual(merged["data"]["time"], [6.0, 7.0, 8.0])
        self.assertEqual(merged["data"]["pt_mean"], [0.2, 0.4, 0.6])
        self.assertEqual(merged["data"]["ft_mean"], [0.1, 0.2, 0.3])

    def test_average_dynamic_time_series_keeps_longer_runs(self) -> None:
        stats = PROCESS_DYNAMIC_GROWTH.average_dynamic_time_series([
            {
                "t_eq_species": 1.0,
                "time": [1.0, 2.0, 3.0],
                "pt": [0.2, 0.4, 0.6],
                "ft": [0.1, 0.2, 0.3],
            },
            {
                "t_eq_species": 1.0,
                "time": [1.0, 2.0, 3.0, 4.0, 5.0],
                "pt": [0.4, 0.6, 0.8, 1.0, 1.2],
                "ft": [0.2, 0.3, 0.4, 0.5, 0.6],
            },
        ])

        self.assertEqual(stats["time"], [1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(stats["pt_N_per_t"], [2, 2, 2, 1, 1])
        self.assertEqual(stats["pt_mean"], [0.30000000000000004, 0.5, 0.7, 1.0, 1.2])
        self.assertEqual(stats["ft_N_per_t"], [2, 2, 2, 1, 1])

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
            self.assertEqual(rows[0]["series"]["C_norm_mean"][0], 0.1)
            self.assertEqual(rows[0]["series"]["r_at_absmax"][1], 3.0)
            self.assertEqual(rows[0]["series"]["t"]["__encoding__"], "range")
            self.assertEqual(rows[0]["series"]["t"]["start"], 10.0)
            self.assertEqual(rows[0]["series"]["t"]["n"], 2)
            self.assertEqual(rows[0]["series"]["n_rows"], 5)

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
            self.assertEqual(bundle["meta"]["format"], "compact_summary_columnar")
            self.assertIsInstance(bundle["samples"][0]["series"], dict)

            loaded_by_file = TIME_SERIES_ANALYSIS.load_lateral_correlations_bundle(out_path)
            loaded_by_dir = TIME_SERIES_ANALYSIS.load_lateral_correlations_bundle(out_dir)
            self.assertEqual(loaded_by_file["meta"]["format"], "compact_summary_columnar")
            self.assertEqual(loaded_by_dir["meta"]["format"], "compact_summary_columnar")

            rows = TIME_SERIES_ANALYSIS.iter_lateral_series_rows(out_path, obs_type="correlation")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["sample_id"], "sample")
            self.assertEqual(rows[0]["t"], 10.0)
            self.assertEqual(rows[0]["C_norm_mean"], 0.1)

    def test_lateral_bundle_merges_new_samples_into_published_mean(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, _, _, data_dir = self._make_data_dir(root)
            correlations_dir = data_dir.parent / "correlations"
            correlations_dir.mkdir(parents=True, exist_ok=True)
            first = correlations_dir / "first_lateral_correlation_time.csv"
            second = correlations_dir / "second_lateral_correlation_time.csv"
            header = "sample_id,dim,L,t,f_t,r_max,n_rows,C_norm_mean,C_norm_std,C_norm_absmax,r_at_absmax,valid_norm_mean,pair_count_mean,boundary_mode"
            first.write_text(
                "\n".join([header, "first,3,8,10,0.2,4,5,0.1,0.02,0.5,2,1,128,periodic"]) + "\n",
                encoding="utf-8",
            )
            second.write_text(
                "\n".join([header, "second,3,8,10,0.2,4,5,0.3,0.04,0.7,4,1,128,periodic"]) + "\n",
                encoding="utf-8",
            )

            out_dir = root / "published"
            out_dir.mkdir()
            out_path = PROCESS_DYNAMIC_GROWTH.process_lateral_correlations(
                data_dir,
                out_dir,
                sample_paths=[first],
            )
            existing_bundle = PROCESS_DYNAMIC_GROWTH.load_lateral_bundle_file(out_path)
            out_path = PROCESS_DYNAMIC_GROWTH.process_lateral_correlations(
                data_dir,
                out_dir,
                sample_paths=[second],
                existing_bundle=existing_bundle,
            )

            with gzip.open(out_path, "rt", encoding="utf-8") as handle:
                bundle = json.load(handle)

            self.assertEqual(bundle["meta"]["aggregation"], "mean_by_parameter")
            self.assertEqual(len(bundle["samples"]), 1)
            self.assertEqual(bundle["samples"][0]["N_samples"], 2)
            rows = TIME_SERIES_ANALYSIS.iter_lateral_series_rows(out_path, obs_type="correlation")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["N_samples"], 2)
            self.assertAlmostEqual(rows[0]["C_norm_mean"], 0.2)
            self.assertAlmostEqual(rows[0]["C_norm_std"], 0.03)

            blocks = TIME_SERIES_ANALYSIS.iter_lateral_series_blocks(out_path, obs_type="correlation")
            self.assertEqual(len(blocks), 1)
            self.assertEqual(blocks[0]["N_samples"], 2)
            self.assertEqual(blocks[0]["series"]["t"], [10.0])
            self.assertAlmostEqual(blocks[0]["series"]["C_norm_mean"][0], 0.2)

            frame = TIME_SERIES_ANALYSIS.load_lateral_correlations_dataframe(out_path, obs_type="correlation")
            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.iloc[0]["N_samples"], 2)
            self.assertAlmostEqual(frame.iloc[0]["C_norm_mean"], 0.2)

    def test_lateral_reader_accepts_legacy_row_series(self) -> None:
        bundle = {
            "samples": [
                {
                    "filename": "sample.csv",
                    "sample_id": "sample",
                    "obs_type": "correlation",
                    "series": [{"t": 1.0, "C_norm_mean": 0.2}],
                }
            ]
        }

        rows = TIME_SERIES_ANALYSIS.iter_lateral_series_rows(bundle, obs_type="correlation")

        self.assertEqual(rows[0]["sample_id"], "sample")
        self.assertEqual(rows[0]["t"], 1.0)
        self.assertEqual(rows[0]["C_norm_mean"], 0.2)


if __name__ == "__main__":
    unittest.main()
