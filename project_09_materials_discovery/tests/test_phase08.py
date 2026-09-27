import unittest

import numpy as np

from p09.graph_model import (GraphTrainingConfig, _batch_graphs,
                             fit_predict_graph_regressor, require_torch,
                             torch_available)
from p09.structure_graph import CrystalGraph
from p09.structure_experiment import _metrics


class Phase08Tests(unittest.TestCase):
    def test_metric_contract(self):
        metrics = _metrics(np.array([1.0, 3.0]), np.array([2.0, 1.0]))
        self.assertEqual(metrics["test_count"], 2)
        self.assertAlmostEqual(metrics["mae"], 1.5)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(2.5))

    @unittest.skipUnless(torch_available(), "install the optional graph extra to test batching")
    def test_disconnected_batch_preserves_graph_membership(self):
        import torch
        first = CrystalGraph(np.array([8, 14]), np.array([[0], [1]]), np.array([1.5]))
        second = CrystalGraph(np.array([6]), np.empty((2, 0), dtype=int), np.empty(0))
        atoms, edges, distances, graph_index = _batch_graphs([first, second], torch.device("cpu"))
        np.testing.assert_array_equal(atoms.numpy(), [8, 14, 6])
        np.testing.assert_array_equal(edges.numpy(), [[0], [1]])
        np.testing.assert_allclose(distances.numpy(), [1.5])
        np.testing.assert_array_equal(graph_index.numpy(), [0, 0, 1])

    def test_optional_runtime_message_or_config(self):
        config = GraphTrainingConfig()
        self.assertEqual(config.cpu_threads, 1)
        self.assertEqual(config.message_passing_steps, 2)
        if not torch_available():
            with self.assertRaisesRegex(RuntimeError, "pip install -e"):
                require_torch()

    @unittest.skipUnless(torch_available(), "install the optional graph extra to test reproducibility")
    def test_fixed_seed_graph_training_is_repeatable(self):
        first = CrystalGraph(np.array([8, 14]), np.array([[0, 1], [1, 0]]), np.array([1.5, 1.5]))
        second = CrystalGraph(np.array([6]), np.empty((2, 0), dtype=int), np.empty(0))
        config = GraphTrainingConfig(hidden_dim=8, embedding_dim=4, radial_basis_size=4,
                                     epochs=2, batch_size=2)
        left = fit_predict_graph_regressor([first, second], [1.0, 2.0], [first, second],
                                           seed=17, config=config)
        right = fit_predict_graph_regressor([first, second], [1.0, 2.0], [first, second],
                                            seed=17, config=config)
        np.testing.assert_array_equal(left, right)


if __name__ == "__main__":
    unittest.main()
