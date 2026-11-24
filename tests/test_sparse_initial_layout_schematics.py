import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


DIM2 = load_module("layout_dim2", ROOT / "tools" / "plot_initial_layout_schematic_dim2.py")
DIM3 = load_module("layout_dim3", ROOT / "tools" / "plot_initial_layout_schematic.py")


class SparseInitialLayoutSchematicTest(unittest.TestCase):
    def test_dim2_blocks_matches_requested_left_gray_right_example(self):
        original = DIM2.L
        DIM2.L = 8
        try:
            first, second = DIM2.sites("blocks", 0.3)
        finally:
            DIM2.L = original
        self.assertEqual(first, [0, 1])
        self.assertEqual(second, [6, 7])

    def test_dim2_alternating_matches_requested_centered_example(self):
        original = DIM2.L
        DIM2.L = 8
        try:
            first, second = DIM2.sites("alternating", 0.3)
        finally:
            DIM2.L = original
        self.assertEqual(first, [1, 4])
        self.assertEqual(second, [2, 5])

    def test_dim2_alternating_spreads_sparse_gray_sites_between_pairs(self):
        first, second = DIM2.sites("alternating", 0.4)
        labels = ["gray"] * DIM2.L
        for i in first:
            labels[i] = "s1"
        for i in second:
            labels[i] = "s2"
        gray_positions = [i for i, label in enumerate(labels) if label == "gray"]
        self.assertEqual(len(gray_positions), 10)
        # At rho=0.4 the gray sites occur throughout the line, rather than
        # forming a single central interval.
        self.assertLess(max(b - a for a, b in zip(gray_positions, gray_positions[1:])), 6)

    def test_dim3_sparse_layouts_have_exact_quotas_and_no_overlap(self):
        original = DIM3.L
        DIM3.L = 4
        try:
            for layout in ("blocks", "alternating"):
                first, second = DIM3.positions(layout, 0.4)
                self.assertEqual(len(first), 6)
                self.assertEqual(len(second), 6)
                self.assertFalse(set(first) & set(second))
        finally:
            DIM3.L = original

    def test_dim3_alternating_uses_opposite_checkerboard_parities(self):
        original = DIM3.L
        DIM3.L = 12
        try:
            first, second = DIM3.positions("alternating", 0.4)
        finally:
            DIM3.L = original
        self.assertTrue(all((x + y) % 2 == 0 for x, y in first))
        self.assertTrue(all((x + y) % 2 == 1 for x, y in second))


if __name__ == "__main__":
    unittest.main()
