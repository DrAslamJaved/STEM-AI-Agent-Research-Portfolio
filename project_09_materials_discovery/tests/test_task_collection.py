import unittest
from unittest.mock import patch

from p09.experiment import load_task


class TaskCollectionTests(unittest.TestCase):
    def test_loader_accepts_matbench_dict_values(self):
        class FakeTask:
            metadata = {"input_type": "composition", "task_type": "regression"}

            def load(self):
                self.loaded = True

        task = FakeTask()

        class FakeBenchmark:
            def __init__(self, **kwargs):
                self.tasks = {"expt_gap": task}.values()

        with patch("matbench.bench.MatbenchBenchmark", FakeBenchmark):
            self.assertIs(load_task(), task)
        self.assertTrue(task.loaded)
