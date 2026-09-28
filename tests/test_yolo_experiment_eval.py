from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ml.yolo_experiment_eval import plan_missing_experiment_evals


class YoloExperimentEvalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.experiments = self.root / "runs" / "yolo_experiments"
        self.eval_root = self.root / "runs" / "eval"
        self.experiments.mkdir(parents=True)
        self.eval_root.mkdir(parents=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_plans_missing_evals_and_skips_existing_folders(self) -> None:
        self._write_best_model(self.experiments / "suite_a" / "no_aug")
        self._write_best_model(self.experiments / "suite_a" / "default")
        (self.eval_root / "suite_a" / "default").mkdir(parents=True)

        jobs, skipped = plan_missing_experiment_evals(
            experiments_dir=self.experiments,
            eval_root=self.eval_root,
            data="./runs/datasets/example/dataset.yaml",
        )

        self.assertEqual(skipped, [self.eval_root / "suite_a" / "default"])
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].experiment_dir, self.experiments / "suite_a" / "no_aug")
        self.assertEqual(jobs[0].model_path, self.experiments / "suite_a" / "no_aug" / "weights" / "best.pt")
        self.assertEqual(jobs[0].eval_dir, self.eval_root / "suite_a" / "no_aug")
        self.assertEqual(jobs[0].output_parent, self.eval_root / "suite_a")
        self.assertEqual(jobs[0].run_name, "no_aug")
        self.assertEqual(jobs[0].data, "./runs/datasets/example/dataset.yaml")

    def test_infers_data_from_experiment_config(self) -> None:
        run_dir = self.experiments / "translation_i1"
        self._write_best_model(run_dir)
        (run_dir / "experiment_config.json").write_text(
            json.dumps({"yolo_args": {"data": "./runs/datasets/from_config/dataset.yaml"}}),
            encoding="utf-8",
        )

        jobs, skipped = plan_missing_experiment_evals(
            experiments_dir=self.experiments,
            eval_root=self.eval_root,
        )

        self.assertEqual(skipped, [])
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].data, "./runs/datasets/from_config/dataset.yaml")

    def test_infers_data_from_parent_manifest(self) -> None:
        suite_dir = self.experiments / "suite_b"
        run_dir = suite_dir / "scale_i1"
        self._write_best_model(run_dir)
        (suite_dir / "experiment_manifest.json").write_text(
            json.dumps({"data": "./runs/datasets/from_manifest/dataset.yaml"}),
            encoding="utf-8",
        )

        jobs, skipped = plan_missing_experiment_evals(
            experiments_dir=self.experiments,
            eval_root=self.eval_root,
        )

        self.assertEqual(skipped, [])
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].data, "./runs/datasets/from_manifest/dataset.yaml")

    def _write_best_model(self, run_dir: Path) -> None:
        weights_dir = run_dir / "weights"
        weights_dir.mkdir(parents=True)
        (weights_dir / "best.pt").write_bytes(b"model")


if __name__ == "__main__":
    unittest.main()
