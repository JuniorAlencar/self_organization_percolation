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

STABILITY_TESTS_PATH = REPO_ROOT / "jupyter" / "src" / "stability_tests.py"
STABILITY_TESTS_SPEC = spec_from_file_location("stability_tests", STABILITY_TESTS_PATH)
STABILITY_TESTS = module_from_spec(STABILITY_TESTS_SPEC)
assert STABILITY_TESTS_SPEC.loader is not None
sys.modules[STABILITY_TESTS_SPEC.name] = STABILITY_TESTS
STABILITY_TESTS_SPEC.loader.exec_module(STABILITY_TESTS)


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

    def _write_sample(
        self,
        data_dir: Path,
        name: str,
        pt: list[float],
        fL_z: list[float] | None = None,
        z_max: float = 1.0,
    ) -> Path:
        data = {
            "color": 1,
            "t_eq_species": 5.0,
            "time": [6.0, 7.0, 8.0],
            "pt": pt,
            "nt": [0.1, 0.2, 0.3],
        }
        if fL_z is not None:
            data["fL_z"] = fL_z
        sample_path = data_dir / name
        sample_path.write_text(
            json.dumps(
                {
                    "meta": {
                        "t_eq_by_species": [5.0],
                        "z_max": [z_max],
                        "z_stat": [z_max],
                        "growth_test_stop_criterion": "alive_species_pt_derivative_stability_or_death",
                        "growth_test_t_eq_validation": "discrete_derivative_of_blocked_pt_variation",
                        "growth_test_t_eq_s_prime_threshold": 1.0e-5,
                        "growth_test_equilibrium_effective_rel_tol": 2.5e-3,
                        "growth_test_post_equilibrium_extra_steps": 100,
                        "growth_test_equilibrium_rel_tol_scaling": "fixed_base_tol_times_0p10",
                    },
                    "results": {
                        "order_percolation 1": {
                            "data": data
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return sample_path

    def _write_empty_sample(self, data_dir: Path, name: str) -> Path:
        sample_path = data_dir / name
        sample_path.write_text(
            json.dumps(
                {
                    "meta": {
                        "t_eq_by_species": [None],
                        "growth_test_stop_criterion": "alive_species_pt_derivative_stability_or_death",
                        "growth_test_t_eq_validation": "discrete_derivative_of_blocked_pt_variation",
                        "growth_test_t_eq_s_prime_threshold": 1.0e-5,
                        "growth_test_equilibrium_effective_rel_tol": 2.5e-3,
                        "growth_test_post_equilibrium_extra_steps": 100,
                    },
                    "results": {},
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
            bundle = PROCESS_DYNAMIC_GROWTH.load_json_bundle(bundle_path)

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

    def test_published_bundle_import_uses_manifest_cache_when_bundle_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root, published_root, manifests_root, data_dir = self._make_data_dir(root)

            self._write_sample(data_dir, "sample_P0_0.7_p0_0.2.json", [0.2, 0.4, 0.6])
            out_path, expected_rows, expected_color_rows = PROCESS_DYNAMIC_GROWTH.process_group(
                data_dir,
                raw_root,
                published_root,
                manifests_root,
                jobs=2,
            )

            rows, color_rows, from_cache = PROCESS_DYNAMIC_GROWTH.rows_from_published_bundle_cached(
                out_path,
                published_root,
                manifests_root,
                series_mode="full",
            )

            self.assertTrue(from_cache)
            self.assertEqual(rows, expected_rows)
            self.assertEqual(color_rows, expected_color_rows)

            manifest = PROCESS_DYNAMIC_GROWTH.load_manifest(
                manifests_root,
                out_path.parent.relative_to(published_root),
            )
            self.assertEqual(
                manifest[PROCESS_DYNAMIC_GROWTH.SUMMARY_FILE_FINGERPRINT_KEY],
                PROCESS_DYNAMIC_GROWTH.file_stat_fingerprint(out_path),
            )

    def test_main_incremental_all_data_updates_only_changed_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, _, _, data_dir = self._make_data_dir(root)
            self._write_sample(data_dir, "sample_P0_0.7_p0_0.2.json", [0.2, 0.4, 0.6])

            old_argv = sys.argv
            original_import = PROCESS_DYNAMIC_GROWTH.rows_from_published_bundle_cached
            try:
                sys.argv = [
                    "process_dynamic_growth.py",
                    "--sop-root",
                    str(root / "SOP_data"),
                    "--series-mode",
                    "full",
                ]
                self.assertEqual(PROCESS_DYNAMIC_GROWTH.main(), 0)

                data_dir_2 = Path(str(data_dir).replace("/L_8/", "/L_16/"))
                data_dir_2.mkdir(parents=True, exist_ok=True)
                self._write_sample(data_dir_2, "sample_P0_0.7_p0_0.2.json", [0.6, 0.8, 1.0])

                def fail_import(*args, **kwargs):
                    raise AssertionError("published bundle import should not run during incremental all-data update")

                PROCESS_DYNAMIC_GROWTH.rows_from_published_bundle_cached = fail_import
                sys.argv = [
                    "process_dynamic_growth.py",
                    "--sop-root",
                    str(root / "SOP_data"),
                    "--series-mode",
                    "full",
                ]
                self.assertEqual(PROCESS_DYNAMIC_GROWTH.main(), 0)
            finally:
                sys.argv = old_argv
                PROCESS_DYNAMIC_GROWTH.rows_from_published_bundle_cached = original_import

            rows = PROCESS_DYNAMIC_GROWTH.read_dat_rows(
                root / "SOP_data" / "all_data_dynamic.dat",
                PROCESS_DYNAMIC_GROWTH.ALL_DATA_COLUMNS,
            )
            self.assertEqual({int(row["L"]) for row in rows}, {8, 16})
            self.assertEqual(len(rows), 2)

    def test_main_incremental_all_data_skips_write_when_no_groups_changed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, _, _, data_dir = self._make_data_dir(root)
            self._write_sample(data_dir, "sample_P0_0.7_p0_0.2.json", [0.2, 0.4, 0.6])

            old_argv = sys.argv
            original_write_all_data = PROCESS_DYNAMIC_GROWTH.write_all_data
            try:
                sys.argv = [
                    "process_dynamic_growth.py",
                    "--sop-root",
                    str(root / "SOP_data"),
                    "--series-mode",
                    "full",
                ]
                self.assertEqual(PROCESS_DYNAMIC_GROWTH.main(), 0)

                def fail_write(*args, **kwargs):
                    raise AssertionError("all_data should not be rewritten when no groups changed")

                PROCESS_DYNAMIC_GROWTH.write_all_data = fail_write
                self.assertEqual(PROCESS_DYNAMIC_GROWTH.main(), 0)
            finally:
                sys.argv = old_argv
                PROCESS_DYNAMIC_GROWTH.write_all_data = original_write_all_data

    def test_main_incremental_all_data_repairs_missing_catalog_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, _, _, data_dir = self._make_data_dir(root)
            self._write_sample(data_dir, "sample_P0_0.7_p0_0.2.json", [0.2, 0.4, 0.6])
            sop_root = root / "SOP_data"
            old_argv = sys.argv
            try:
                sys.argv = [
                    "process_dynamic_growth.py",
                    "--sop-root",
                    str(sop_root),
                    "--series-mode",
                    "full",
                ]
                self.assertEqual(PROCESS_DYNAMIC_GROWTH.main(), 0)
                PROCESS_DYNAMIC_GROWTH.write_all_data([], sop_root / "all_data_dynamic.dat")

                self.assertEqual(PROCESS_DYNAMIC_GROWTH.main(), 0)
            finally:
                sys.argv = old_argv

            rows = PROCESS_DYNAMIC_GROWTH.read_dat_rows(
                sop_root / "all_data_dynamic.dat",
                PROCESS_DYNAMIC_GROWTH.ALL_DATA_COLUMNS,
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["control_rule"], "relative")
            self.assertEqual(rows[0]["stat_window"], 0)

    def test_incremental_merge_updates_total_samples_for_orders_not_in_new_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root, published_root, manifests_root, data_dir = self._make_data_dir(root)

            first_path = self._write_sample(data_dir, "sample_a_P0_0.7_p0_0.2.json", [0.2, 0.4, 0.6])
            first_data = json.loads(first_path.read_text(encoding="utf-8"))
            first_data["results"]["order_percolation 2"] = json.loads(
                json.dumps(first_data["results"]["order_percolation 1"])
            )
            first_data["results"]["order_percolation 2"]["data"]["pt"] = [0.6, 0.8, 1.0]
            first_path.write_text(json.dumps(first_data), encoding="utf-8")

            PROCESS_DYNAMIC_GROWTH.process_group(
                data_dir,
                raw_root,
                published_root,
                manifests_root,
                jobs=2,
            )

            self._write_sample(data_dir, "sample_c_P0_0.7_p0_0.2.json", [0.8, 1.0, 1.2])
            out_path, all_rows, all_color_rows = PROCESS_DYNAMIC_GROWTH.process_group(
                data_dir,
                raw_root,
                published_root,
                manifests_root,
                jobs=2,
            )

            bundle = PROCESS_DYNAMIC_GROWTH.load_json_bundle(out_path)
            group = bundle["p0_groups"][0]
            orders = {order["order"]: order for order in group["orders"]}

            self.assertEqual(group["num_samples_total"], 2)
            self.assertEqual(orders[0]["N_samples"], 2)
            self.assertEqual(orders[0]["N_samples_perc"], 2)
            self.assertEqual(orders[1]["N_samples"], 2)
            self.assertEqual(orders[1]["N_samples_perc"], 1)
            self.assertEqual(orders[0]["data"]["n_samples_total"], 2)
            self.assertEqual(orders[1]["data"]["n_samples_total"], 2)
            self.assertEqual({row["order"]: row["N_samples"] for row in all_rows}, {0: 2, 1: 2})
            self.assertEqual({row["order"]: row["N_samples_perc"] for row in all_rows}, {0: 2, 1: 1})
            self.assertEqual(all_color_rows[0]["N_samples"], 2)

    def test_incremental_merge_counts_metadata_only_samples_only_in_total_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root, published_root, manifests_root, data_dir = self._make_data_dir(root)

            self._write_sample(data_dir, "sample_a_P0_0.7_p0_0.2.json", [0.2, 0.4, 0.6])
            PROCESS_DYNAMIC_GROWTH.process_group(
                data_dir,
                raw_root,
                published_root,
                manifests_root,
                jobs=2,
            )

            self._write_empty_sample(data_dir, "sample_empty_P0_0.7_p0_0.2.json")
            out_path, all_rows, all_color_rows = PROCESS_DYNAMIC_GROWTH.process_group(
                data_dir,
                raw_root,
                published_root,
                manifests_root,
                jobs=2,
            )

            bundle = PROCESS_DYNAMIC_GROWTH.load_json_bundle(out_path)
            group = bundle["p0_groups"][0]
            order = group["orders"][0]

            self.assertEqual(group["num_samples_total"], 2)
            self.assertEqual(order["N_samples"], 2)
            self.assertEqual(order["N_samples_perc"], 1)
            self.assertEqual(order["data"]["n_samples_total"], 2)
            self.assertEqual(order["data"]["n_samples_perc"], 1)
            self.assertEqual(all_rows[0]["N_samples"], 2)
            self.assertEqual(all_rows[0]["N_samples_perc"], 1)
            self.assertEqual(all_color_rows[0]["N_samples"], 2)
            self.assertAlmostEqual(all_color_rows[0]["nc"], 0.5)

    def test_new_group_with_only_metadata_only_samples_counts_total_but_has_no_order_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root, published_root, manifests_root, data_dir = self._make_data_dir(root)

            self._write_empty_sample(data_dir, "sample_empty_P0_0.7_p0_0.2.json")
            out_path, all_rows, all_color_rows = PROCESS_DYNAMIC_GROWTH.process_group(
                data_dir,
                raw_root,
                published_root,
                manifests_root,
                jobs=2,
            )

            bundle = PROCESS_DYNAMIC_GROWTH.load_json_bundle(out_path)
            group = bundle["p0_groups"][0]

            self.assertEqual(group["num_samples_total"], 1)
            self.assertEqual(group["orders"], [])
            self.assertEqual(all_rows, [])
            self.assertEqual(all_color_rows[0]["N_samples"], 1)
            self.assertEqual(all_color_rows[0]["nc"], 0.0)

    def test_invalid_zero_byte_samples_are_ignored_in_sample_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root, published_root, manifests_root, data_dir = self._make_data_dir(root)

            self._write_sample(data_dir, "sample_a_P0_0.7_p0_0.2.json", [0.2, 0.4, 0.6])
            (data_dir / "sample_empty_0kb_P0_0.7_p0_0.2.json").write_text("", encoding="utf-8")
            out_path, all_rows, all_color_rows = PROCESS_DYNAMIC_GROWTH.process_group(
                data_dir,
                raw_root,
                published_root,
                manifests_root,
                jobs=2,
            )

            bundle = PROCESS_DYNAMIC_GROWTH.load_json_bundle(out_path)
            group = bundle["p0_groups"][0]
            order = group["orders"][0]

            self.assertEqual(group["num_samples_total"], 1)
            self.assertEqual(order["N_samples"], 1)
            self.assertEqual(order["N_samples_perc"], 1)
            self.assertEqual(all_rows[0]["N_samples"], 1)
            self.assertEqual(all_rows[0]["N_samples_perc"], 1)
            self.assertEqual(all_color_rows[0]["N_samples"], 1)

    def test_updates_time_series_from_published_when_raw_is_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root, published_root, manifests_root, data_dir = self._make_data_dir(root)
            sample_name = "sample_P0_0.7_p0_0.2.json"

            self._write_sample(data_dir, sample_name, [0.2, 0.4, 0.6])
            out_path, all_rows, _ = PROCESS_DYNAMIC_GROWTH.process_group(
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
                detect_replaced_files=True,
            )

            self.assertEqual(all_rows[0]["N_samples"], 2)
            bundle = PROCESS_DYNAMIC_GROWTH.load_json_bundle(out_path)
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

    def test_merge_aligns_incremental_series_by_time_and_sample_count(self) -> None:
        old = {
            "order": 0,
            "N_samples": 2,
            "N_samples_perc": 2,
            "data": {
                "time": [1.0, 2.0],
                "pt_mean": [2.0, 4.0],
                "pt_std": [1.0, 2.0],
                "pt_N_per_t": [2, 2],
                "n_seeds_pt": 2,
                "n_seeds_ft": 0,
            },
            "p": {"mean": 3.0, "err": 0.0, "n": 2},
            "f": {"mean": None, "err": None, "n": 0},
            "t_eq_species": {"mean": 1.0, "err": 0.0, "n": 2},
            "z_stat": {"mean": 1.0, "err": 0.0, "n": 2},
            "samples": [],
        }
        new = {
            "order": 0,
            "N_samples": 1,
            "N_samples_perc": 1,
            "data": {
                "time": [2.0, 3.0],
                "pt_mean": [10.0, 20.0],
                "pt_std": [0.0, 0.0],
                "pt_N_per_t": [1, 1],
                "n_seeds_pt": 1,
                "n_seeds_ft": 0,
            },
            "p": {"mean": 15.0, "err": 0.0, "n": 1},
            "f": {"mean": None, "err": None, "n": 0},
            "t_eq_species": {"mean": 1.0, "err": 0.0, "n": 1},
            "z_stat": {"mean": 1.0, "err": 0.0, "n": 1},
            "samples": [],
        }

        data = PROCESS_DYNAMIC_GROWTH.merge_order_block(old, new)["data"]

        self.assertEqual(data["time"], [1.0, 2.0, 3.0])
        self.assertEqual(data["pt_N_per_t"], [2, 3, 1])
        self.assertEqual(data["pt_mean"], [2.0, 6.0, 20.0])
        self.assertAlmostEqual(data["pt_std"][1], 14.0 ** 0.5)
        self.assertEqual(data["pt_common_time"], [2.0])

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
        self.assertEqual(stats["pt_common_time"], [1.0, 2.0, 3.0])
        self.assertEqual(stats["pt_common_mean"], [0.30000000000000004, 0.5, 0.7])
        self.assertEqual(stats["pt_supported_time"], [1.0, 2.0, 3.0])
        self.assertEqual(stats["pt_supported_mean"], [0.30000000000000004, 0.5, 0.7])
        self.assertEqual(stats["pt_min_support_count"], 2)
        self.assertEqual(stats["ft_N_per_t"], [2, 2, 2, 1, 1])

    def test_average_dynamic_time_series_adds_flz_height_mean(self) -> None:
        stats = PROCESS_DYNAMIC_GROWTH.average_dynamic_time_series([
            {
                "t_eq_species": 1.0,
                "time": [1.0, 2.0, 3.0],
                "pt": [0.2, 0.4, 0.6],
                "ft": [0.1, 0.2, 0.3],
                "fL_z": [0.25, 0.5, 0.0],
            },
            {
                "t_eq_species": 1.0,
                "time": [1.0, 2.0, 3.0],
                "pt": [0.4, 0.6, 0.8],
                "ft": [0.2, 0.3, 0.4],
                "fL_z": [0.5, 0.25],
            },
        ])

        self.assertEqual(stats["fL_z_z"], [0, 1, 2])
        self.assertEqual(stats["fL_z_N_per_z"], [2, 2, 1])
        self.assertEqual(stats["fL_z_mean"], [0.375, 0.375, 0.0])
        self.assertEqual(stats["fL_z_common_z"], [0, 1])
        self.assertEqual(stats["fL_z_common_mean"], [0.375, 0.375])
        self.assertEqual(stats["fL_z_supported_z"], [0, 1])
        self.assertEqual(stats["fL_z_supported_mean"], [0.375, 0.375])

    def test_process_group_writes_flz_height_mean(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root, published_root, manifests_root, data_dir = self._make_data_dir(root)
            self._write_sample(data_dir, "sample_a_P0_0.7_p0_0.2.json", [0.2, 0.4, 0.6], [0.25, 0.5, 0.0])
            self._write_sample(data_dir, "sample_b_P0_0.7_p0_0.2.json", [0.6, 0.8, 1.0], [0.5, 0.25])

            out_path, all_rows, _ = PROCESS_DYNAMIC_GROWTH.process_group(
                data_dir,
                raw_root,
                published_root,
                manifests_root,
                jobs=2,
            )

            bundle = PROCESS_DYNAMIC_GROWTH.load_json_bundle(out_path)
            order_data = bundle["p0_groups"][0]["orders"][0]["data"]
            self.assertEqual(order_data["fL_z_z"], [0, 1, 2])
            self.assertEqual(order_data["fL_z_N_per_z"], [2, 2, 1])
            self.assertEqual(order_data["fL_z_mean"], [0.375, 0.375, 0.0])
            self.assertEqual(order_data["fL_z_supported_z"], [0, 1])
            self.assertEqual(order_data["fL_z_supported_mean"], [0.375, 0.375])

    def test_process_group_writes_zmax_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root, published_root, manifests_root, data_dir = self._make_data_dir(root)
            self._write_sample(data_dir, "sample_a_P0_0.7_p0_0.2.json", [0.2, 0.4, 0.6], z_max=3.0)
            self._write_sample(data_dir, "sample_b_P0_0.7_p0_0.2.json", [0.6, 0.8, 1.0], z_max=5.0)

            out_path, all_rows, _ = PROCESS_DYNAMIC_GROWTH.process_group(
                data_dir,
                raw_root,
                published_root,
                manifests_root,
                jobs=2,
            )

            bundle = PROCESS_DYNAMIC_GROWTH.load_json_bundle(out_path)
            order = bundle["p0_groups"][0]["orders"][0]
            self.assertEqual(order["z_max"]["values"], [3.0, 5.0])
            self.assertEqual(order["z_max"]["median"], 4.0)
            self.assertEqual(order["z_max"]["q75"], 4.5)
            self.assertEqual(order["z_max"]["q90"], 4.8)
            self.assertEqual(order["data"]["z_max_values"], [3.0, 5.0])
            self.assertEqual(order["data"]["z_max_median"], 4.0)
            self.assertEqual(order["data"]["z_max_q75"], 4.5)
            self.assertEqual(order["data"]["z_max_q90"], 4.8)
            self.assertAlmostEqual(order["data"]["z_max_mean"], 4.0)
            self.assertEqual(all_rows[0]["z_max_median"], 4.0)
            self.assertEqual(all_rows[0]["z_max_q75"], 4.5)
            self.assertEqual(all_rows[0]["z_max_q90"], 4.8)

    def test_all_data_header_uses_zmax_quantiles_without_removed_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "all_data_dynamic.dat"
            rows = [
                {
                    "type_perc": "S1",
                    "dim": 2,
                    "L": 8,
                    "f_T": 0.3,
                    "c": 0.2,
                    "nc": 2,
                    "rho": 0.2,
                    "p0": 0.2,
                    "P0": 0.7,
                    "order": 1,
                    "N_samples": 2,
                    "N_samples_perc": 2,
                    "p_mean": 0.6,
                    "p_err": 0.2,
                    "f_mean": 0.15,
                    "f_err": 0.05,
                    "z_max_mean": 4.0,
                    "z_max_err": 1.0,
                    "z_max_median": 4.0,
                    "z_max_q75": 4.5,
                    "z_max_q90": 4.8,
                    "stat_window": 0,
                    "stop_criterion": "old_header_field",
                    "equilibrium_effective_rel_tol": 2.5e-3,
                    "post_equilibrium_extra_steps": 100,
                    "t_eq_validation": "validation",
                    "t_eq_s_prime_threshold": 1.0e-5,
                    "z_stat_mean": 4.0,
                    "z_stat_err": 1.0,
                }
            ]

            PROCESS_DYNAMIC_GROWTH.write_all_data(rows, output_path)

            lines = output_path.read_text(encoding="utf-8").splitlines()
            header = lines[0].split()
            data = lines[1].split()

            self.assertIn("z_max_median", header)
            self.assertIn("z_max_q75", header)
            self.assertIn("z_max_q90", header)
            self.assertNotIn("z_stat_mean", header)
            self.assertNotIn("z_stat_err", header)
            self.assertNotIn("stat_window", header)
            self.assertNotIn("stop_criterion", header)
            self.assertNotIn("equilibrium_effective_rel_tol", header)
            self.assertNotIn("post_equilibrium_extra_steps", header)
            self.assertEqual(data[header.index("z_max_median")], "4")
            self.assertEqual(data[header.index("z_max_q75")], "4.5")
            self.assertEqual(data[header.index("z_max_q90")], "4.8")

    def test_process_group_writes_per_sample_p_tail_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root, published_root, manifests_root, data_dir = self._make_data_dir(root)
            self._write_sample(data_dir, "sample_a_P0_0.7_p0_0.2.json", [0.2, 0.4, 0.6])
            self._write_sample(data_dir, "sample_b_P0_0.7_p0_0.2.json", [0.6, 0.8, 1.0])

            out_path, _, _ = PROCESS_DYNAMIC_GROWTH.process_group(
                data_dir,
                raw_root,
                published_root,
                manifests_root,
                jobs=2,
            )

            bundle = PROCESS_DYNAMIC_GROWTH.load_json_bundle(out_path)
            data = bundle["p0_groups"][0]["orders"][0]["data"]
            self.assertEqual(len(data["p_tail_sample_values"]), 2)
            self.assertAlmostEqual(data["p_tail_sample_values"][0], 0.4)
            self.assertAlmostEqual(data["p_tail_sample_values"][1], 0.8)
            self.assertEqual(
                data["p_tail_estimator"],
                "mean_of_per_sample_tail_means_after_each_sample_t_eq",
            )
            self.assertAlmostEqual(data["p_tail_mean"], 0.6)

    def test_process_group_profiles_mode_skips_time_series_but_keeps_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root, published_root, manifests_root, data_dir = self._make_data_dir(root)
            self._write_sample(data_dir, "sample_a_P0_0.7_p0_0.2.json", [0.2, 0.4, 0.6], [0.25, 0.5])

            out_path, _, _ = PROCESS_DYNAMIC_GROWTH.process_group(
                data_dir,
                raw_root,
                published_root,
                manifests_root,
                jobs=2,
                series_mode="profiles",
            )

            bundle = PROCESS_DYNAMIC_GROWTH.load_json_bundle(out_path)
            data = bundle["p0_groups"][0]["orders"][0]["data"]
            self.assertEqual(bundle["meta"]["series_mode"], "profiles")
            self.assertEqual(data["pt_mean"], [])
            self.assertEqual(data["fL_z_mean"], [0.25, 0.5])
            self.assertAlmostEqual(data["p_tail_mean"], 0.4)

    def test_incremental_processing_separates_feedback_control_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root, published_root, manifests_root, data_dir = self._make_data_dir(root)
            self._write_sample(data_dir, "relative_P0_0.7_p0_0.2.json", [0.2, 0.4, 0.6])
            out_path, _, _ = PROCESS_DYNAMIC_GROWTH.process_group(
                data_dir,
                raw_root,
                published_root,
                manifests_root,
                jobs=1,
            )

            linear_path = self._write_sample(
                data_dir,
                "linear_P0_0.7_p0_0.2.json",
                [0.6, 0.8, 1.0],
            )
            linear_sample = json.loads(linear_path.read_text(encoding="utf-8"))
            linear_sample["meta"]["feedback_control_rule"] = "linear"
            linear_path.write_text(json.dumps(linear_sample), encoding="utf-8")

            out_path, rows, color_rows = PROCESS_DYNAMIC_GROWTH.process_group(
                data_dir,
                raw_root,
                published_root,
                manifests_root,
                jobs=1,
            )

            bundle = PROCESS_DYNAMIC_GROWTH.load_json_bundle(out_path)
            groups = bundle["p0_groups"]
            self.assertEqual(bundle["meta"]["feedback_control_rule"], ["linear", "relative"])
            self.assertEqual(
                {group["feedback_control_rule"] for group in groups},
                {"linear", "relative"},
            )
            self.assertTrue(all(group["num_samples_total"] == 1 for group in groups))
            self.assertEqual({row["control_rule"] for row in rows}, {"linear", "relative"})
            self.assertEqual({row["control_rule"] for row in color_rows}, {"linear", "relative"})

    def test_same_filename_from_different_raw_roots_is_processed_twice(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, published_root, manifests_root, relative_dir = self._make_data_dir(root)
            rel_group = PROCESS_DYNAMIC_GROWTH.canonical_rel_group(relative_dir)
            linear_dir = root / "SOP_data" / "tests_data" / "linear" / rel_group / "data"
            linear_dir.mkdir(parents=True)
            filename = "same_seed_P0_0.7_p0_0.2.json"
            self._write_sample(relative_dir, filename, [0.2, 0.4, 0.6])
            linear_path = self._write_sample(linear_dir, filename, [0.6, 0.8, 1.0])
            linear_sample = json.loads(linear_path.read_text(encoding="utf-8"))
            linear_sample["meta"]["feedback_control_rule"] = "linear"
            linear_path.write_text(json.dumps(linear_sample), encoding="utf-8")

            out_path, rows, _ = PROCESS_DYNAMIC_GROWTH.process_group(
                rel_group,
                [relative_dir, linear_dir],
                published_root,
                manifests_root,
                jobs=1,
            )

            bundle = PROCESS_DYNAMIC_GROWTH.load_json_bundle(out_path)
            manifest = PROCESS_DYNAMIC_GROWTH.load_manifest(manifests_root, rel_group)
            self.assertEqual(
                {group["feedback_control_rule"] for group in bundle["p0_groups"]},
                {"linear", "relative"},
            )
            self.assertEqual({row["control_rule"] for row in rows}, {"linear", "relative"})
            self.assertEqual(len(manifest["processed_json_files"]), 2)
            self.assertTrue(all("::" in key for key in manifest["processed_json_files"]))
            self.assertNotEqual(
                PROCESS_DYNAMIC_GROWTH.sample_cache_path(Path("cache"), relative_dir / filename),
                PROCESS_DYNAMIC_GROWTH.sample_cache_path(Path("cache"), linear_dir / filename),
            )

    def test_fallback_group_keys_include_control_rule(self) -> None:
        params = {
            "type_perc": "bond",
            "dim": 2,
            "L": 8,
            "f_T": 0.3,
            "c": 0.1,
            "nc": 1,
            "rho": 1.0,
            "p0": 0.8,
            "P0": 0.2,
            "stat_window": 0,
        }

        relative = PROCESS_DYNAMIC_GROWTH.all_data_group_key_from_params(params, "relative")
        linear = PROCESS_DYNAMIC_GROWTH.all_data_group_key_from_params(params, "linear")

        self.assertNotEqual(relative, linear)
        self.assertIn("relative", relative)
        self.assertIn("linear", linear)

    def test_legacy_manifest_collision_is_resolved_by_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, _, _, relative_dir = self._make_data_dir(root)
            rel_group = PROCESS_DYNAMIC_GROWTH.canonical_rel_group(relative_dir)
            linear_dir = root / "SOP_data" / "tests_data" / "linear" / rel_group / "data"
            linear_dir.mkdir(parents=True)
            filename = "same_seed_P0_0.7_p0_0.2.json"
            relative_path = self._write_sample(relative_dir, filename, [0.2, 0.4, 0.6])
            linear_path = self._write_sample(linear_dir, filename, [0.6, 0.8, 1.0])
            relative_id = PROCESS_DYNAMIC_GROWTH.sample_manifest_id(relative_path)
            linear_id = PROCESS_DYNAMIC_GROWTH.sample_manifest_id(linear_path)
            old_fingerprint = PROCESS_DYNAMIC_GROWTH.file_fingerprint_for_mode(
                relative_path, "stat"
            )

            files, fingerprints = PROCESS_DYNAMIC_GROWTH.migrate_legacy_manifest_file_ids(
                {filename},
                {filename: old_fingerprint},
                {relative_id: relative_path, linear_id: linear_path},
                "stat",
            )

            self.assertEqual(files, {relative_id})
            self.assertEqual(fingerprints, {relative_id: old_fingerprint})

    def test_legacy_bundle_control_rule_defaults_to_relative(self) -> None:
        bundle = {
            "meta": {"series_mode": "full"},
            "p0_groups": [{"P0_value": 0.2, "p0_value": 0.8, "orders": []}],
        }

        changed = PROCESS_DYNAMIC_GROWTH.update_dynamic_bundle_control_rules(bundle)

        self.assertTrue(changed)
        self.assertEqual(bundle["meta"]["feedback_control_rule"], "relative")
        self.assertEqual(bundle["p0_groups"][0]["feedback_control_rule"], "relative")

    def test_read_dynamic_bundle_exposes_series_profiles_and_heights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "properties_dynamic_bundle.json"
            bundle_path.write_text(
                json.dumps(
                    {
                        "meta": {
                            "type_perc": "S1",
                            "dim": 2,
                            "L": 8,
                            "f_T": 0.3,
                            "c": 0.2,
                            "nc": 2,
                            "rho": 0.2,
                            "stat_window": 0,
                            "series_mode": "full",
                        },
                        "p0_groups": [
                            {
                                "P0_value": 0.7,
                                "p0_value": 0.2,
                                "num_samples_total": 2,
                                "colors": {"nc": 1.5, "nc_err": 0.1, "nc_std": 0.2},
                                "orders": [
                                    {
                                        "order": 1,
                                        "N_samples_perc": 2,
                                        "data": {
                                            "time": [1.0, 2.0, 3.0],
                                            "pt_mean": [0.2, 0.4, 0.6],
                                            "pt_N_per_t": [2, 2, 1],
                                            "ft_time": [1.0, 2.0],
                                            "ft_mean": [0.1, 0.2],
                                            "ft_N_per_t": [2, 1],
                                            "fL_z_z": [0, 1, 2],
                                            "fL_z_mean": [0.25, 0.5, 0.0],
                                            "fL_z_N_per_z": [2, 2, 1],
                                            "fL_z_supported_z": [0, 1],
                                            "fL_z_supported_mean": [0.25, 0.5],
                                            "z_max_median": 4.0,
                                            "z_max_q75": 4.5,
                                            "z_max_q90": 4.8,
                                            "z_max_values": [3.0, 5.0],
                                            "p_tail_sample_values": [0.4, 0.8],
                                        },
                                        "p": {"mean": 0.6, "err": 0.2},
                                        "f": {"mean": 0.15, "err": 0.05},
                                        "t_eq_species": {"mean": 5.0, "err": 0.0},
                                        "z_max": {"mean": 4.0, "err": 1.0, "std": 1.0},
                                        "z_stat": {"mean": 4.0, "err": 1.0, "std": 1.0},
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            frame = STABILITY_TESTS.read_dynamic_bundle(bundle_path)
            row = frame.iloc[0]

            self.assertEqual(row["series_mode"], "full")
            self.assertEqual(row["pt_N_per_t"], [2, 2, 1])
            self.assertEqual(row["ft_time"], [1.0, 2.0])
            self.assertEqual(row["fL_z_z"], [0, 1, 2])
            self.assertEqual(row["fL_z_mean"], [0.25, 0.5, 0.0])
            self.assertEqual(row["fL_z_supported_z"], [0, 1])
            self.assertEqual(row["z_max_median"], 4.0)
            self.assertEqual(row["z_max_q75"], 4.5)
            self.assertEqual(row["z_max_q90"], 4.8)
            self.assertEqual(row["z_max_values"], [3.0, 5.0])
            self.assertEqual(row["p_tail_sample_values"], [0.4, 0.8])

    def test_compress_published_only_migrates_without_raw(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            published_root = root / "SOP_data" / "published_dynamic"
            group_dir = published_root / "S1_percolation" / "num_colors_2"
            group_dir.mkdir(parents=True)
            legacy_dynamic = group_dir / "properties_dynamic_bundle.json"
            legacy_dynamic.write_text(
                json.dumps({
                    "meta": {"type_perc": "S1", "dim": 2, "L": 8, "f_T": 0.3, "c": 0.2, "nc": 2, "rho": 0.2},
                    "p0_groups": [],
                }),
                encoding="utf-8",
            )
            dynamic_n, _ = PROCESS_DYNAMIC_GROWTH.compress_published_only(
                published_root,
                root / "SOP_data",
                "all_data_dynamic.dat",
                "all_colors_dynamic.dat",
                compresslevel=1,
                threads=1,
            )

            self.assertEqual(dynamic_n, 1)
            self.assertFalse(legacy_dynamic.exists())
            self.assertTrue((group_dir / "properties_dynamic_bundle.json.xz").exists())
            bundle = PROCESS_DYNAMIC_GROWTH.load_json_bundle(group_dir / "properties_dynamic_bundle.json.xz")
            self.assertEqual(bundle["meta"]["L"], 8)

    def test_compress_published_only_updates_zmax_q90_from_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            published_root = root / "SOP_data" / "published_dynamic"
            group_dir = published_root / "S1_percolation" / "num_colors_2"
            group_dir.mkdir(parents=True)
            legacy_dynamic = group_dir / "properties_dynamic_bundle.json"
            legacy_dynamic.write_text(
                json.dumps(
                    {
                        "meta": {
                            "type_perc": "S1",
                            "dim": 2,
                            "L": 8,
                            "f_T": 0.3,
                            "c": 0.2,
                            "nc": 2,
                            "rho": 0.2,
                        },
                        "p0_groups": [
                            {
                                "P0_value": 0.7,
                                "p0_value": 0.2,
                                "num_samples_total": 4,
                                "orders": [
                                    {
                                        "order": 1,
                                        "N_samples": 4,
                                        "N_samples_perc": 4,
                                        "data": {"z_max_values": [1.0, 3.0, 5.0, 9.0]},
                                        "z_max": {"values": [1.0, 3.0, 5.0, 9.0]},
                                        "p": {},
                                        "f": {},
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            dynamic_n, _ = PROCESS_DYNAMIC_GROWTH.compress_published_only(
                published_root,
                root / "SOP_data",
                "all_data_dynamic.dat",
                "all_colors_dynamic.dat",
                compresslevel=1,
                threads=1,
                rebuild_all_data=True,
            )

            self.assertEqual(dynamic_n, 1)
            bundle = PROCESS_DYNAMIC_GROWTH.load_json_bundle(group_dir / "properties_dynamic_bundle.json.xz")
            order = bundle["p0_groups"][0]["orders"][0]
            self.assertAlmostEqual(order["z_max"]["q90"], 7.8)
            self.assertAlmostEqual(order["data"]["z_max_q90"], 7.8)

            all_data = root / "SOP_data" / "all_data_dynamic.dat"
            lines = all_data.read_text(encoding="utf-8").splitlines()
            header = lines[0].split()
            data = lines[1].split()
            self.assertIn("z_max_q90", header)
            self.assertEqual(data[header.index("z_max_q90")], "7.8")

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

if __name__ == "__main__":
    unittest.main()
