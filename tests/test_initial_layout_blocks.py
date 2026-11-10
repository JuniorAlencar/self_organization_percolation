import json
import tempfile
import unittest
from pathlib import Path
from importlib.util import module_from_spec, spec_from_file_location

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "process_dynamic_growth.py"
SPEC = spec_from_file_location("process_dynamic_growth", MODULE_PATH)
PROCESS_DYNAMIC_GROWTH = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PROCESS_DYNAMIC_GROWTH)


class InitialLayoutBlocksTest(unittest.TestCase):
    def _make_data_dir(self, root: Path) -> tuple[Path, Path, Path, Path]:
        raw_root = root / "SOP_data" / "raw_growth_test_dynamic"
        published_root = root / "SOP_data" / "published_dynamic"
        manifests_root = root / "SOP_data" / "manifests_dynamic"
        data_dir = (
            raw_root
            / "bond_percolation"
            / "num_colors_1"
            / "dim_2"
            / "L_100"
            / "fT_constant"
            / "fT_0.4"
            / "c_0.05"
            / "rho_1.0"
            / "data"
        )
        data_dir.mkdir(parents=True, exist_ok=True)
        return raw_root, published_root, manifests_root, data_dir

    def _write_sample(
        self,
        data_dir: Path,
        name: str,
        pt: list[float],
        control_rule: str = "linear",
        layout: str | None = None,
    ) -> Path:
        out_path = data_dir / name
        payload = {
            "meta": {
                "feedback_control_rule": control_rule,
                "t_eq_by_species": [1.0],
                "z_stat": [100.0],
            },
            "results": {
                "order_percolation 1": {
                    "data": {
                        "color": 1,
                        "t_eq_species": 1.0,
                        "time": [0.0, 1.0, 2.0],
                        "pt": pt,
                        "nt": [0.4, 0.4, 0.4],
                    }
                }
            },
        }
        if layout:
            payload["meta"]["initial_layout"] = layout
            payload["meta"]["initial_base_layout"] = layout
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        return out_path

    def test_samples_with_different_layouts_are_grouped_into_separate_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root, published_root, manifests_root, data_dir = self._make_data_dir(root)

            # Two samples with same P0 and p0, but different layouts in filename
            self._write_sample(data_dir, "sample_1_P0_0.20_p0_0.80_base_clustered.json", [0.5, 0.5, 0.5])
            self._write_sample(data_dir, "sample_2_P0_0.20_p0_0.80_base_alternating.json", [0.6, 0.6, 0.6])
            # A third sample with base_random (or default)
            self._write_sample(data_dir, "sample_3_P0_0.20_p0_0.80.json", [0.7, 0.7, 0.7])

            out_path, _, _ = PROCESS_DYNAMIC_GROWTH.process_group(
                data_dir, raw_root, published_root, manifests_root, jobs=1
            )

            bundle = PROCESS_DYNAMIC_GROWTH.load_json_bundle(out_path)
            groups = bundle.get("p0_groups", [])

            # Must have 3 distinct blocks!
            self.assertEqual(len(groups), 3)

            groups_by_layout = {g["initial_layout"]: g for g in groups}
            self.assertIn("clustered", groups_by_layout)
            self.assertIn("alternating", groups_by_layout)
            self.assertIn("random", groups_by_layout)

            for layout, g in groups_by_layout.items():
                self.assertEqual(g["num_samples_total"], 1)
                self.assertEqual(g["P0_value"], 0.2)
                self.assertEqual(g["p0_value"], 0.8)
                self.assertEqual(g["orders"][0]["samples"][0]["initial_layout"], layout)

    def test_incremental_merge_updates_same_layout_and_creates_new_block_for_new_layout_or_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root, published_root, manifests_root, data_dir = self._make_data_dir(root)

            # Step 1: Initial batch with 1 clustered sample at P0=0.2, p0=0.8
            self._write_sample(data_dir, "sample_1_P0_0.20_p0_0.80_base_clustered.json", [0.5, 0.5, 0.5])
            out_path, _, _ = PROCESS_DYNAMIC_GROWTH.process_group(
                data_dir, raw_root, published_root, manifests_root, jobs=1
            )
            bundle = PROCESS_DYNAMIC_GROWTH.load_json_bundle(out_path)
            self.assertEqual(len(bundle["p0_groups"]), 1)
            self.assertEqual(bundle["p0_groups"][0]["initial_layout"], "clustered")
            self.assertEqual(bundle["p0_groups"][0]["num_samples_total"], 1)

            # Step 2: Incremental update with:
            # - sample_2: same layout (clustered), same P0=0.2, p0=0.8 -> should UPDATE existing block!
            # - sample_3: different layout (alternating), same P0=0.2, p0=0.8 -> should CREATE NEW block!
            # - sample_4: distinct P0=0.3, p0=0.7 (clustered) -> should CREATE NEW block!
            self._write_sample(data_dir, "sample_2_P0_0.20_p0_0.80_base_clustered.json", [0.52, 0.52, 0.52])
            self._write_sample(data_dir, "sample_3_P0_0.20_p0_0.80_base_alternating.json", [0.6, 0.6, 0.6])
            self._write_sample(data_dir, "sample_4_P0_0.30_p0_0.70_base_clustered.json", [0.4, 0.4, 0.4])

            out_path, _, _ = PROCESS_DYNAMIC_GROWTH.process_group(
                data_dir, raw_root, published_root, manifests_root, jobs=1
            )
            bundle = PROCESS_DYNAMIC_GROWTH.load_json_bundle(out_path)
            groups = bundle.get("p0_groups", [])

            # Must now have 3 blocks total:
            # 1. (clustered, 0.2, 0.8) -> updated to 2 samples
            # 2. (alternating, 0.2, 0.8) -> 1 sample
            # 3. (clustered, 0.3, 0.7) -> 1 sample
            self.assertEqual(len(groups), 3)

            key_map = {
                (g["initial_layout"], round(float(g["P0_value"]), 2), round(float(g["p0_value"]), 2)): g
                for g in groups
            }

            self.assertIn(("clustered", 0.2, 0.8), key_map)
            self.assertIn(("alternating", 0.2, 0.8), key_map)
            self.assertIn(("clustered", 0.3, 0.7), key_map)

            # Verify block update:
            clustered_02_08 = key_map[("clustered", 0.2, 0.8)]
            self.assertEqual(clustered_02_08["num_samples_total"], 2)
            self.assertEqual(len(clustered_02_08["orders"][0]["samples"]), 2)

            # Verify new block creation:
            alternating_02_08 = key_map[("alternating", 0.2, 0.8)]
            self.assertEqual(alternating_02_08["num_samples_total"], 1)

            clustered_03_07 = key_map[("clustered", 0.3, 0.7)]
            self.assertEqual(clustered_03_07["num_samples_total"], 1)

    def test_split_p0_group_by_layout_separates_legacy_mixed_group(self) -> None:
        mixed_group = {
            "feedback_control_rule": "linear",
            "initial_layout": ["alternating", "random"],
            "P0_value": 0.2,
            "p0_value": 0.8,
            "num_samples_total": 3,
            "orders": [
                {
                    "order": 0,
                    "N_samples": 3,
                    "data": {"time": [0, 1]},
                    "samples": [
                        {"filename": "s1_base_alternating.json", "initial_layout": "alternating", "p_sample_mean": 0.5, "f_sample_mean": 0.4, "z_stat": 100.0, "t_eq_species": 1.0},
                        {"filename": "s2_base_alternating.json", "initial_layout": "alternating", "p_sample_mean": 0.52, "f_sample_mean": 0.4, "z_stat": 100.0, "t_eq_species": 1.0},
                        {"filename": "s3_base_random.json", "initial_layout": "random", "p_sample_mean": 0.6, "f_sample_mean": 0.4, "z_stat": 100.0, "t_eq_species": 1.0},
                    ],
                }
            ],
        }

        split = PROCESS_DYNAMIC_GROWTH.split_p0_group_by_layout(mixed_group)
        self.assertEqual(len(split), 2)
        by_layout = {g["initial_layout"]: g for g in split}
        self.assertIn("alternating", by_layout)
        self.assertIn("random", by_layout)

        self.assertEqual(by_layout["alternating"]["num_samples_total"], 2)
        self.assertEqual(len(by_layout["alternating"]["orders"][0]["samples"]), 2)
        self.assertEqual(by_layout["random"]["num_samples_total"], 1)
        self.assertEqual(len(by_layout["random"]["orders"][0]["samples"]), 1)

    def test_dat_columns_and_read_write_include_initial_layout(self) -> None:
        self.assertIn("initial_layout", PROCESS_DYNAMIC_GROWTH.ALL_DATA_COLUMNS)
        self.assertIn("initial_layout", PROCESS_DYNAMIC_GROWTH.ALL_COLORS_COLUMNS)
        self.assertIn("initial_layout", PROCESS_DYNAMIC_GROWTH.ALL_DATA_GROUP_COLUMNS)
        self.assertIn("initial_layout", PROCESS_DYNAMIC_GROWTH.ALL_COLORS_GROUP_COLUMNS)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dat = Path(tmpdir) / "test_data.dat"
            sample_rows = [
                {
                    "type_perc": "bond", "dim": 2, "L": 100, "f_T": 0.1, "c": 0.05,
                    "nc": 1, "rho": 1.0, "p0": 0.8, "P0": 0.2, "control_rule": "linear",
                    "initial_layout": "clustered", "stat_window": 0, "order": 0,
                    "N_samples": 10, "N_samples_perc": 10, "p_mean": 0.5, "p_err": 0.01,
                    "f_mean": 0.1, "f_err": 0.01, "z_stat_mean": 50.0, "z_stat_err": 1.0,
                    "z_stat_median": 50.0, "z_stat_q75": 55.0, "z_stat_q90": 60.0,
                    "t_eq_validation": "none", "t_eq_s_prime_threshold": 1e-5,
                }
            ]
            PROCESS_DYNAMIC_GROWTH.write_all_data(sample_rows, out_dat)
            loaded_rows = PROCESS_DYNAMIC_GROWTH.read_dat_rows(out_dat, PROCESS_DYNAMIC_GROWTH.ALL_DATA_COLUMNS)
            self.assertEqual(len(loaded_rows), 1)
            self.assertEqual(loaded_rows[0]["initial_layout"], "clustered")

    def test_stability_tests_bundle_functions_support_initial_layout(self) -> None:
        import sys
        sys.path.insert(0, str(REPO_ROOT / "jupyter"))
        from src.stability_tests import read_dynamic_bundle, select_dynamic_bundle_row

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test_bundle.json"
            bundle_data = {
                "meta": {
                    "type_perc": "bond", "dim": 2, "L": 100, "f_T": 0.1, "c": 0.05,
                    "nc": 1, "rho": 1.0, "stat_window": 0, "series_mode": "full",
                    "feedback_control_rule": "linear",
                },
                "p0_groups": [
                    {
                        "P0_value": 0.2,
                        "p0_value": 0.8,
                        "initial_layout": "alternating",
                        "num_samples_total": 5,
                        "colors": {"nc": 1.0},
                        "orders": [
                            {"order": 0, "N_samples_perc": 5, "p": {"mean": 0.505}, "f": {"mean": 0.1}, "z_stat": {}, "z_max": {}, "data": {}}
                        ],
                    },
                    {
                        "P0_value": 0.2,
                        "p0_value": 0.8,
                        "initial_layout": "random",
                        "num_samples_total": 3,
                        "colors": {"nc": 1.0},
                        "orders": [
                            {"order": 0, "N_samples_perc": 3, "p": {"mean": 0.510}, "f": {"mean": 0.1}, "z_stat": {}, "z_max": {}, "data": {}}
                        ],
                    },
                ],
            }
            bundle_path.write_text(json.dumps(bundle_data), encoding="utf-8")

            # Default initial_layout='alternating'
            df_alt = read_dynamic_bundle(bundle_path)
            self.assertEqual(len(df_alt), 1)
            self.assertEqual(df_alt.iloc[0]["initial_layout"], "alternating")
            row_alt = select_dynamic_bundle_row(df_alt, p0=0.8, P0=0.2, order=0, control_rule="linear")
            self.assertEqual(row_alt["initial_layout"], "alternating")

            # Explicit initial_layout='random'
            df_rnd = read_dynamic_bundle(bundle_path, initial_layout="random")
            self.assertEqual(len(df_rnd), 1)
            self.assertEqual(df_rnd.iloc[0]["initial_layout"], "random")
            row_rnd = select_dynamic_bundle_row(df_rnd, p0=0.8, P0=0.2, order=0, control_rule="linear", initial_layout="random")
            self.assertEqual(row_rnd["initial_layout"], "random")

            # All layouts with initial_layout='all'
            df_all = read_dynamic_bundle(bundle_path, initial_layout="all")
            self.assertEqual(len(df_all), 2)
            row_alt2 = select_dynamic_bundle_row(df_all, p0=0.8, P0=0.2, order=0, control_rule="linear", initial_layout="alternating")
            self.assertEqual(row_alt2["initial_layout"], "alternating")
            row_rnd2 = select_dynamic_bundle_row(df_all, p0=0.8, P0=0.2, order=0, control_rule="linear", InitialLayout="random")
            self.assertEqual(row_rnd2["initial_layout"], "random")

    def test_colors_analysis_get_ft_min_max_includes_initial_layout(self) -> None:
        import sys
        import pandas as pd
        sys.path.insert(0, str(REPO_ROOT / "jupyter"))
        from src.ColorsAnalysis import get_ft_min_max

        # Provide a synthetic df_series
        df_series = pd.DataFrame([
            {
                "type_perc": "bond", "dim": 2, "L": 512, "f_T": 0.2, "c": 0.05,
                "nc": 1, "rho": 1.0, "p0": 0.8, "P0": 0.2, "control_rule": "linear",
                "initial_layout": "alternating", "order": 0, "N_samples": 100,
                "N_samples_perc": 100, "p_mean": 0.505, "p_err": 0.001,
            }
        ])
        res = get_ft_min_max(
            df_series=df_series,
            L_lst=[512],
            c_lst=[0.05],
            type_perc_lst=["bond"],
            from_plots=False,
        )
        self.assertIn("initial_layout", res.columns)
        self.assertEqual(res.iloc[0]["initial_layout"], "alternating")


if __name__ == "__main__":
    unittest.main()
