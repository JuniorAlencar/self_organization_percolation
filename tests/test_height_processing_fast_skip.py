import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import process_height_ensemble_series as HEIGHT_ENSEMBLE  # noqa: E402
import process_height_timeseries as HEIGHT_SAMPLES  # noqa: E402


class HeightProcessingFastSkipTest(unittest.TestCase):
    def _paths(self, root: Path) -> tuple[Path, Path, Path]:
        raw_root = root / "SOP_data" / "raw_growth_test_dynamic"
        data_dir = (
            raw_root
            / "bond_percolation"
            / "num_colors_1"
            / "dim_2"
            / "L_8"
            / "fT_constant"
            / "fT_0.1"
            / "c_0.1"
            / "rho_1.0"
            / "data"
        )
        data_dir.mkdir(parents=True)
        out_root = root / "SOP_data" / "processed_height_timeseries"
        return raw_root, data_dir, out_root

    def test_sample_processor_adopts_outputs_then_skips_entire_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_root, data_dir, out_root = self._paths(Path(tmpdir))
            out_dir = HEIGHT_SAMPLES.output_group_dir_from_data_dir(raw_root, out_root, data_dir)
            out_dir.mkdir(parents=True)
            (out_dir / HEIGHT_SAMPLES.HEIGHT_SAMPLE_FILE).write_bytes(b"group")
            (out_root / "height_sample_measures_all.csv.gz").write_bytes(b"all")
            (out_root / "height_group_summary.csv.gz").write_bytes(b"summary")
            manifest = {
                "height_sample_processing_version": HEIGHT_SAMPLES.HEIGHT_SAMPLE_PROCESSING_VERSION,
                HEIGHT_SAMPLES.DATA_DIR_FINGERPRINT_KEY:
                    HEIGHT_SAMPLES.directory_stat_fingerprint(data_dir),
                HEIGHT_SAMPLES.HEIGHT_SAMPLE_SUMMARY_CACHE_KEY: [],
            }
            (out_dir / HEIGHT_SAMPLES.HEIGHT_SAMPLE_MANIFEST).write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            old_argv = sys.argv
            try:
                sys.argv = ["process_height_timeseries.py", "--root", str(raw_root), "--out-root", str(out_root)]
                self.assertEqual(HEIGHT_SAMPLES.main(), 0)
                self.assertEqual(HEIGHT_SAMPLES.main(), 0)
            finally:
                sys.argv = old_argv

            state_path = HEIGHT_SAMPLES.height_run_state_path(out_root, raw_root, "height_samples")
            self.assertTrue(state_path.exists())
            self.assertEqual((out_root / "height_sample_measures_all.csv.gz").read_bytes(), b"all")

    def test_ensemble_processor_adopts_outputs_then_skips_entire_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_root, data_dir, out_root = self._paths(Path(tmpdir))
            out_dir = HEIGHT_SAMPLES.output_group_dir_from_data_dir(raw_root, out_root, data_dir)
            out_dir.mkdir(parents=True)
            (out_dir / HEIGHT_ENSEMBLE.HEIGHT_ENSEMBLE_FILE).write_bytes(b"group")
            (out_root / "height_ensemble_timeseries_all.csv.gz").write_bytes(b"all")
            manifest = {
                "height_ensemble_processing_version": HEIGHT_ENSEMBLE.HEIGHT_ENSEMBLE_PROCESSING_VERSION,
                HEIGHT_SAMPLES.DATA_DIR_FINGERPRINT_KEY:
                    HEIGHT_SAMPLES.directory_stat_fingerprint(data_dir),
            }
            (out_dir / HEIGHT_ENSEMBLE.HEIGHT_ENSEMBLE_MANIFEST).write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            old_argv = sys.argv
            try:
                sys.argv = ["process_height_ensemble_series.py", "--root", str(raw_root), "--out-root", str(out_root)]
                self.assertEqual(HEIGHT_ENSEMBLE.main(), 0)
                self.assertEqual(HEIGHT_ENSEMBLE.main(), 0)
            finally:
                sys.argv = old_argv

            state_path = HEIGHT_SAMPLES.height_run_state_path(out_root, raw_root, "height_ensemble")
            self.assertTrue(state_path.exists())
            self.assertEqual((out_root / "height_ensemble_timeseries_all.csv.gz").read_bytes(), b"all")

    def test_run_state_changes_when_data_directory_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_root, data_dir, _ = self._paths(Path(tmpdir))
            data_dirs = HEIGHT_SAMPLES.collect_data_dirs(
                raw_root,
                type("Args", (), {"type_perc": None, "lengths": None, "f_T": None})(),
            )
            before = HEIGHT_SAMPLES.build_height_run_state(data_dirs, 1, {})
            (data_dir / "new.yts").write_bytes(b"new")
            after = HEIGHT_SAMPLES.build_height_run_state(data_dirs, 1, {})

            self.assertNotEqual(before, after)


if __name__ == "__main__":
    unittest.main()
