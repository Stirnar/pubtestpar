from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

from ml.yolo_training import (
    NO_AUGMENTATION_PARAMS,
    _epoch_from_row,
    _write_tensorboard_scalars,
    build_yolo_experiment_plan,
)

TENSORBOARD_AVAILABLE = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("tensorboard") is not None
)


class YoloTrainingExperimentTests(unittest.TestCase):
    def test_builds_default_augmentation_experiment_plan(self) -> None:
        experiments = build_yolo_experiment_plan(max_intensity=3)
        names = [experiment.name for experiment in experiments]

        self.assertEqual(len(experiments), 20)
        self.assertEqual(names[:2], ["no_aug", "default"])
        self.assertIn("translation_i3", names)
        self.assertIn("scale_i3", names)
        self.assertIn("rotation_i3", names)
        self.assertIn("flip_i3", names)
        self.assertIn("brightness_contrast_i3", names)
        self.assertIn("hsv_color_i3", names)

        no_aug = experiments[0]
        default = experiments[1]
        translation = next(experiment for experiment in experiments if experiment.name == "translation_i3")

        self.assertEqual(no_aug.augmentation, NO_AUGMENTATION_PARAMS)
        self.assertEqual(default.augmentation, {})
        self.assertEqual(translation.augmentation["translate"], 0.20)
        self.assertEqual(translation.augmentation["scale"], 0.0)
        self.assertEqual(translation.augmentation["fliplr"], 0.0)

    def test_builds_aggressive_augmentation_experiment_plan(self) -> None:
        experiments = build_yolo_experiment_plan(max_intensity=4, grid_preset="aggressive")
        names = [experiment.name for experiment in experiments]

        self.assertEqual(len(names), len(set(names)))
        self.assertIn("translation_i4", names)
        self.assertIn("hsv_color_i4", names)
        self.assertIn("shear_i4", names)
        self.assertIn("perspective_i4", names)
        self.assertIn("mosaic_i4", names)
        self.assertIn("mixup_i4", names)
        self.assertIn("ood_color_i4", names)
        self.assertIn("ood_geometry_i4", names)
        self.assertIn("ood_acquisition_i4", names)

        hsv_color = next(experiment for experiment in experiments if experiment.name == "hsv_color_i4")
        acquisition = next(experiment for experiment in experiments if experiment.name == "ood_acquisition_i4")

        self.assertGreater(hsv_color.augmentation["hsv_h"], 0.015)
        self.assertGreater(hsv_color.augmentation["hsv_s"], 0.70)
        self.assertGreater(hsv_color.augmentation["hsv_v"], 0.40)
        self.assertEqual(acquisition.augmentation["mosaic"], 1.0)
        self.assertGreater(acquisition.augmentation["mixup"], 0.0)
        self.assertGreater(acquisition.augmentation["translate"], 0.10)

    def test_builds_compact_mixed_augmentation_grid(self) -> None:
        experiments = build_yolo_experiment_plan(max_intensity=3, grid_preset="compact")
        names = [experiment.name for experiment in experiments]

        self.assertEqual(
            names,
            [
                "no_aug",
                "default",
                "hsv_color_i4",
                "scale_i4",
                "ood_color_i2",
                "ood_color_i4",
                "ood_geometry_i2",
                "ood_geometry_i4",
                "ood_acquisition_i2",
                "ood_acquisition_i3",
                "ood_acquisition_i4",
                "mosaic_mix_i2",
                "mosaic_mix_i4",
            ],
        )
        acquisition = next(experiment for experiment in experiments if experiment.name == "ood_acquisition_i4")
        self.assertGreater(acquisition.augmentation["hsv_v"], 0.40)
        self.assertGreater(acquisition.augmentation["scale"], 0.50)
        self.assertEqual(acquisition.augmentation["mosaic"], 1.0)

    def test_compact_grid_rejects_strategy_filter(self) -> None:
        with self.assertRaises(ValueError):
            build_yolo_experiment_plan(
                max_intensity=4,
                grid_preset="compact",
                strategies=["rotation"],
            )

    def test_plan_can_run_subset_with_prefix(self) -> None:
        experiments = build_yolo_experiment_plan(
            max_intensity=2,
            strategies=["rotation"],
            include_default=False,
            include_no_augmentation=False,
            run_prefix="ascaris",
        )

        self.assertEqual([experiment.name for experiment in experiments], ["ascaris_rotation_i1", "ascaris_rotation_i2"])
        self.assertEqual([experiment.augmentation["degrees"] for experiment in experiments], [5.0, 10.0])

    def test_max_intensity_is_capped_at_four(self) -> None:
        with self.assertRaises(ValueError):
            build_yolo_experiment_plan(max_intensity=5)

    def test_epoch_parser_handles_ultralytics_spaced_csv_headers(self) -> None:
        self.assertEqual(_epoch_from_row({"                  epoch": "2"}, fallback=0), 2)

    @unittest.skipUnless(TENSORBOARD_AVAILABLE, "torch and tensorboard are required.")
    def test_writes_tensorboard_events_from_results_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            results_csv = run_dir / "results.csv"
            with results_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["                  epoch", "         train/box_loss"])
                writer.writeheader()
                writer.writerow({"                  epoch": "0", "         train/box_loss": "1.2"})
                writer.writerow({"                  epoch": "1", "         train/box_loss": "0.8"})

            tensorboard_dir = _write_tensorboard_scalars(run_dir)

            self.assertEqual(tensorboard_dir, run_dir / "tensorboard")
            self.assertTrue(list(tensorboard_dir.glob("events.out.tfevents.*")))


if __name__ == "__main__":
    unittest.main()
