from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ml.yolo_prediction_bundle import PredictionBundle, _boxes_from_result, _normalize_names


class _FakeBoxes:
    xyxy = [[10, 20, 30, 40], [50, 60, 80, 100]]
    conf = [0.9, 0.4]
    cls = [1, 0]


class _FakeResult:
    boxes = _FakeBoxes()


class YoloPredictionBundleTests(unittest.TestCase):
    def test_boxes_from_result_uses_xyxy_conf_and_cls(self) -> None:
        boxes = _boxes_from_result(_FakeResult(), ["ascaris_lumbricoides", "trichuria_trichuris"])

        self.assertEqual(len(boxes), 2)
        self.assertEqual(boxes[0]["bbox"], [10.0, 20.0, 20.0, 20.0])
        self.assertEqual(boxes[0]["label"], "trichuria_trichuris")
        self.assertEqual(boxes[0]["confidence"], 0.9)
        self.assertEqual(boxes[1]["bbox"], [50.0, 60.0, 30.0, 40.0])
        self.assertEqual(boxes[1]["label"], "ascaris_lumbricoides")

    def test_normalize_names_accepts_dict(self) -> None:
        self.assertEqual(_normalize_names({1: "trichuria", 0: "ascaris"}), ["ascaris", "trichuria"])

    def test_prediction_bundle_filters_by_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / "images" / "a.jpg").write_bytes(b"fake")
            (root / "predictions_viewer.json").write_text(
                json.dumps(
                    {
                        "classes": ["ascaris_lumbricoides"],
                        "images": [
                            {
                                "id": 1,
                                "source_filename": "a.jpg",
                                "display_filename": "a.jpg",
                                "width": 100,
                                "height": 100,
                            }
                        ],
                        "annotations": [
                            {
                                "id": 1,
                                "image_id": 1,
                                "class_id": 0,
                                "label": "ascaris_lumbricoides",
                                "confidence": 0.9,
                                "bbox": [1, 2, 3, 4],
                            },
                            {
                                "id": 2,
                                "image_id": 1,
                                "class_id": 0,
                                "label": "ascaris_lumbricoides",
                                "confidence": 0.1,
                                "bbox": [5, 6, 7, 8],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            bundle = PredictionBundle(root)

            self.assertEqual(bundle.list_images(0.25)[0]["visible_boxes"], 1)
            self.assertEqual(len(bundle.boxes_for(1, 0.25)), 1)
            self.assertEqual(bundle.stats(0.25)["visible_predictions"], 1)


if __name__ == "__main__":
    unittest.main()
