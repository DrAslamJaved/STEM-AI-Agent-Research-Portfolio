import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from p09.structure_graph import CrystalGraph, crystal_graph, structure_descriptors
from p09.structure_task import audit_fold, load_structure_task


class Neighbour:
    def __init__(self, index, distance, image=(0, 0, 0)):
        self.index, self.nn_distance, self.image = index, distance, image


class Lattice:
    abc = (3., 4., 5.)
    angles = (90., 90., 120.)


class Structure:
    atomic_numbers = [8, 14, 8]
    density = 2.5
    volume = 60.
    lattice = Lattice()

    def get_all_neighbors(self, radius):
        if radius != 5.:
            raise ValueError("expected cutoff")
        return [[Neighbour(2, 2.0), Neighbour(1, 1.0), Neighbour(1, 3.0)],
                [Neighbour(0, 1.0)], []]


class StructureTask:
    folds = [0]
    metadata = {"input_type": "structure", "task_type": "regression"}

    def load(self):
        self.loaded = True

    def get_train_and_val_data(self, fold):
        if fold != 0 or isinstance(fold, str):
            raise TypeError("fold must be integer")
        return [Structure(), Structure(), Structure()], np.arange(3.)

    def get_test_data(self, fold, include_target=False):
        if fold != 0 or include_target:
            raise TypeError("test graph audit must not request targets")
        return [Structure(), Structure()]


class Phase07Tests(unittest.TestCase):
    def test_graph_truncation_and_descriptor_shape(self):
        graph = crystal_graph(Structure(), 5., max_neighbors=2)
        np.testing.assert_array_equal(graph.edge_index, [[0, 0, 1], [1, 2, 0]])
        np.testing.assert_allclose(graph.edge_distance, [1., 2., 1.])
        self.assertEqual(len(structure_descriptors(Structure())), 129)
        self.assertEqual(crystal_graph(Structure(), 5., max_neighbors=1).edge_index.shape[1], 2)
        with self.assertRaisesRegex(ValueError, "positive"):
            crystal_graph(Structure(), 0.)

    def test_invalid_graph_and_target_free_audit(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            CrystalGraph(np.array([1]), np.array([[0], [0]]), np.array([0.])).validate()
        audit = audit_fold(StructureTask(), 0, limit=2)
        self.assertEqual(audit["train"]["records"], 3)
        self.assertEqual(audit["test"]["records"], 2)
        self.assertEqual(audit["test"]["descriptor_dimension"], 129)

    def test_structure_loader_accepts_task_values(self):
        task = StructureTask()
        class Benchmark:
            def __init__(self, autoload, subset):
                self.tasks = {"dielectric": task}.values()
        with patch.dict(sys.modules, {"matbench": types.ModuleType("matbench"),
                                      "matbench.bench": types.SimpleNamespace(MatbenchBenchmark=Benchmark)}):
            self.assertIs(load_structure_task(), task)
        self.assertTrue(task.loaded)


if __name__ == "__main__":
    unittest.main()
