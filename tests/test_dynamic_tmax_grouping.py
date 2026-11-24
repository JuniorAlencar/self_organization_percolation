import json
import subprocess
import tempfile
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "process_dynamic_growth.py"
SPEC = spec_from_file_location("process_dynamic_growth", MODULE_PATH)
PROCESS_DYNAMIC_GROWTH = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PROCESS_DYNAMIC_GROWTH)


class DynamicTmaxGroupingTest(unittest.TestCase):
    def _make_sop_root(self, root: Path) -> tuple[Path, Path, Path, Path, Path]:
        sop_root = root / "SOP_data"
        raw_root = sop_root / "tests_data" / "linear"
        published_root = sop_root / "published_dynamic"
        manifests_root = sop_root / "manifests_dynamic"
        data_dir = (
            raw_root
            / "bond_percolation"
            / "num_colors_1"
            / "dim_2"
            / "L_512"
            / "fT_constant"
            / "fT_0.15"
            / "c_0.05"
            / "rho_1.0"
            / "data"
        )
        data_dir.mkdir(parents=True, exist_ok=True)
        published_root.mkdir(parents=True, exist_ok=True)
        manifests_root.mkdir(parents=True, exist_ok=True)
        return sop_root, raw_root, published_root, manifests_root, data_dir

    def _write_sample(
        self,
        data_dir: Path,
        name: str,
        pt: list[float] | None,
        *,
        stop_reason: str | None = None,
        stop_time: int | None = None,
        control_rule: str = "linear",
    ) -> Path:
        meta = {
            "num_colors": 1,
            "feedback_control_rule": control_rule,
            "initial_base_layout": "alternating",
        }
        if stop_reason is not None:
            meta["growth_test_stop_reason_actual"] = stop_reason
        if stop_time is not None:
            meta["growth_test_stop_time"] = stop_time

        if pt is None:
            # Dead run
            meta["t_eq_by_species"] = [None]
            meta["z_stat"] = [None]
            results = {}
        else:
            meta["t_eq_by_species"] = [1.0]
            meta["z_stat"] = [120]
            results = {
                "order_percolation 1": {
                    "data": {
                        "color": 1,
                        "t_eq_species": 1.0,
                        "time": [0.0, 1.0, 2.0, 3.0],
                        "pt": pt,
                        "nt": [0.15, 0.15, 0.15, 0.15],
                    }
                }
            }
        out = data_dir / name
        out.write_text(json.dumps({"meta": meta, "results": results}), encoding="utf-8")
        return out

    def test_existing_published_bundles_migrate_to_free_tmax_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sop_root, _, published_root, manifests_root, _ = self._make_sop_root(Path(tmpdir))
            rel_group = Path("bond_percolation/num_colors_1/dim_2/L_512/fT_constant/fT_0.15/c_0.05/rho_1.0")
            b_dir = published_root / rel_group
            b_dir.mkdir(parents=True, exist_ok=True)
            b_path = b_dir / "properties_dynamic_bundle.json.xz"

            legacy_bundle = {
                "meta": {
                    "type_perc": "bond", "dim": 2, "L": 512, "f_T": 0.15, "c": 0.05,
                    "nc": 1, "rho": 1.0, "stat_window": 0, "series_mode": "full",
                    "feedback_control_rule": "linear", "initial_layout": "alternating",
                },
                "p0_groups": [
                    {
                        "feedback_control_rule": "linear",
                        "initial_layout": "alternating",
                        "initial_base_layout": "alternating",
                        "P0_value": 0.2,
                        "p0_value": 0.8,
                        "num_samples_total": 10,
                        "colors": {"Nsamples": 10, "nc": 1.0, "nc_err": 0.0, "nc_std": 0.0},
                        "orders": [
                            {
                                "order": 0,
                                "N_samples": 10,
                                "N_samples_perc": 10,
                                "p": {"mean": 0.502, "err": 0.001},
                                "f": {"mean": 0.15, "err": 0.001},
                                "z_stat": {"mean": 120.0, "err": 1.0, "median": 120.0, "q75": 125.0, "q90": 130.0},
                                "samples": [{"filename": "old_1_P0_0.20_p0_0.80_base_alternating.json", "color": 1}],
                            }
                        ],
                    }
                ],
            }
            PROCESS_DYNAMIC_GROWTH.write_json_bundle(b_path, legacy_bundle)

            updated = PROCESS_DYNAMIC_GROWTH.migrate_published_tmax_once(
                published_root, manifests_root, jobs=1
            )
            self.assertEqual(updated, 1)

            migrated = PROCESS_DYNAMIC_GROWTH.load_json_bundle(b_path)
            self.assertEqual(migrated["meta"]["TMAX"], "free")
            self.assertEqual(migrated["meta"]["growth_test_stop_time"], 0)
            pg = migrated["p0_groups"][0]
            self.assertEqual(pg["TMAX"], "free")
            self.assertEqual(pg["growth_test_stop_time"], 0)
            self.assertEqual(pg["orders"][0]["samples"][0]["TMAX"], "free")
            self.assertEqual(pg["orders"][0]["samples"][0]["growth_test_stop_time"], 0)

            rows, color_rows = PROCESS_DYNAMIC_GROWTH.rows_from_bundle(migrated)
            self.assertEqual(rows[0]["TMAX"], 0)
            self.assertEqual(color_rows[0]["TMAX"], 0)

    def test_alignment_and_grouping_separates_free_and_numeric_tmax_and_aligns_dead_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sop_root, _, published_root, manifests_root, data_dir = self._make_sop_root(Path(tmpdir))

            # Batch 1 on machine 'apolo2' (free run: post_equilibrium_height_reached at variable times + 1 dead)
            self._write_sample(
                data_dir,
                "apolo2_seed_1_ts_20261124T001000_P0_0.20_p0_0.80_base_alternating.json",
                [0.50, 0.50, 0.50, 0.50],
                stop_reason="post_equilibrium_height_reached",
                stop_time=430,
            )
            self._write_sample(
                data_dir,
                "apolo2_seed_2_ts_20261124T001005_P0_0.20_p0_0.80_base_alternating.json",
                [0.52, 0.52, 0.52, 0.52],
                stop_reason="post_equilibrium_height_reached",
                stop_time=580,
            )
            self._write_sample(
                data_dir,
                "apolo2_seed_3_ts_20261124T001010_P0_0.20_p0_0.80_base_alternating.json",
                None,
                stop_reason="all_dead",
                stop_time=19,
            )

            # Batch 2 on machine 'germano2' (fixed TMAX = 1500: hard_max_steps at 1500 + 1 dead at t=14 + subcritical p0=0.20 all_dead)
            self._write_sample(
                data_dir,
                "germano2_seed_10_ts_20261124T020000_P0_0.20_p0_0.80_base_alternating.json",
                [0.60, 0.60, 0.60, 0.60],
                stop_reason="hard_max_steps",
                stop_time=1500,
            )
            self._write_sample(
                data_dir,
                "germano2_seed_11_ts_20261124T020005_P0_0.20_p0_0.80_base_alternating.json",
                None,
                stop_reason="all_dead",
                stop_time=14,
            )
            self._write_sample(
                data_dir,
                "germano2_seed_12_ts_20261124T020010_P0_0.20_p0_0.20_base_alternating.json",
                None,
                stop_reason="all_dead",
                stop_time=8,
            )

            # Run full tools/update_dynamic.sh end-to-end on this SOP_data root!
            proc = subprocess.run(
                [
                    "bash",
                    str(REPO_ROOT / "tools" / "update_dynamic.sh"),
                    "--sop-root",
                    str(sop_root),
                    "-j",
                    "2",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("[update_dynamic] All tasks finished successfully.", proc.stdout)

            bundles = list(published_root.rglob("properties_dynamic_bundle.json.xz"))
            self.assertEqual(len(bundles), 1)
            bundle = PROCESS_DYNAMIC_GROWTH.load_json_bundle(bundles[0])
            self.assertEqual(bundle["meta"]["TMAX"], ["free", 1500])
            self.assertEqual(bundle["meta"]["growth_test_stop_time"], [0, 1500])

            by_key = {
                (g["growth_test_stop_time"], round(float(g["p0_value"]), 2)): g
                for g in bundle["p0_groups"]
            }
            # Must have:
            # - (0, 0.80): 3 total samples (2 perc, 1 dead), TMAX='free', p_mean=0.51
            # - (1500, 0.80): 2 total samples (1 perc, 1 dead), TMAX=1500, p_mean=0.60
            # - (1500, 0.20): 1 total sample (0 perc, 1 dead), TMAX=1500 (aligned from germano2!)
            self.assertIn((0, 0.80), by_key)
            self.assertIn((1500, 0.80), by_key)
            self.assertIn((1500, 0.20), by_key)

            g_free = by_key[(0, 0.80)]
            self.assertEqual(g_free["TMAX"], "free")
            self.assertEqual(g_free["num_samples_total"], 3)
            self.assertEqual(g_free["orders"][0]["N_samples_perc"], 2)
            self.assertAlmostEqual(g_free["orders"][0]["p"]["mean"], 0.51)

            g_1500 = by_key[(1500, 0.80)]
            self.assertEqual(g_1500["TMAX"], 1500)
            self.assertEqual(g_1500["num_samples_total"], 2)
            self.assertEqual(g_1500["orders"][0]["N_samples_perc"], 1)
            self.assertAlmostEqual(g_1500["orders"][0]["p"]["mean"], 0.60)

            g_dead_sub = by_key[(1500, 0.20)]
            self.assertEqual(g_dead_sub["TMAX"], 1500)
            self.assertEqual(g_dead_sub["num_samples_total"], 1)
            self.assertEqual(g_dead_sub["orders"], [])

            # Check all_data_dynamic.dat and all_colors_dynamic.dat TMAX column (0 when free, 1500 when numeric)
            all_data_rows = PROCESS_DYNAMIC_GROWTH.read_dat_rows(
                sop_root / "all_data_dynamic.dat", PROCESS_DYNAMIC_GROWTH.ALL_DATA_COLUMNS
            )
            all_colors_rows = PROCESS_DYNAMIC_GROWTH.read_dat_rows(
                sop_root / "all_colors_dynamic.dat", PROCESS_DYNAMIC_GROWTH.ALL_COLORS_COLUMNS
            )
            self.assertEqual(sorted(r["TMAX"] for r in all_data_rows), [0, 1500])
            self.assertEqual(sorted(r["TMAX"] for r in all_colors_rows), [0, 1500, 1500])

    def test_all_dead_sibling_ft_group_uses_nearby_hard_stop_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, _, _, target_dir = self._make_sop_root(Path(tmpdir))
            sibling_dir = target_dir.parents[3] / "fT_0.16" / target_dir.relative_to(target_dir.parents[2])
            sibling_dir.mkdir(parents=True)

            target = self._write_sample(
                target_dir,
                "light_seed_1_ts_20261124T152500_P0_0.20_p0_0.80_base_alternating.json",
                None,
                stop_reason="all_dead",
                stop_time=48,
            )
            self._write_sample(
                sibling_dir,
                "light_seed_2_ts_20261124T152501_P0_0.20_p0_0.80_base_alternating.json",
                [0.5, 0.5, 0.5, 0.5],
                stop_reason="hard_max_steps",
                stop_time=1500,
            )

            aligned = PROCESS_DYNAMIC_GROWTH.align_tmax_across_files([target])
            self.assertEqual(aligned[target][2], 1500)


if __name__ == "__main__":
    unittest.main()
