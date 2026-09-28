from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from external_dataset_audit import external_dataset_audit_csv
from external_datasets import (
    build_external_review_bundle,
    convert_external_xml_annotations,
    discover_external_datasets,
    external_counts_csv,
    external_label_normalization_report,
    label_from_image_filename,
    normalize_external_label,
)


class ExternalDatasetTests(unittest.TestCase):
    def test_counts_coco_csv_and_yolo_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_coco_dataset(root)
            self._write_csv_dataset(root)
            self._write_image_json_dataset(root)
            self._write_yolo_dataset(root)

            rows = list(csv.DictReader(external_counts_csv(root).splitlines()))
            by_dataset = {row["dataset"]: row for row in rows}

            self.assertEqual(by_dataset["coco_public"]["ascaris_lumbricoides"], "2")
            self.assertEqual(by_dataset["csv_public"]["taenia_spp"], "1")
            self.assertEqual(by_dataset["image_json_public"]["ascaris_lumbricoides"], "1")
            self.assertEqual(by_dataset["image_json_public"]["hookworm"], "1")
            self.assertEqual(by_dataset["image_json_public"]["schistosoma_mansoni"], "1")
            self.assertEqual(by_dataset["image_json_public"]["trichuris_trichiura"], "1")
            self.assertEqual(by_dataset["yolo_public"]["trichuris_trichiura"], "2")

    def test_external_dataset_audit_reports_inventory_and_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_coco_dataset(root)
            self._write_image_json_dataset(root)
            self._write_voc_xml_dataset(root, stem="valid")
            self._write_voc_xml_dataset(
                root,
                stem="missing",
                filename="does_not_exist.jpg",
                write_image=False,
            )
            self._write_yolo_data_yaml_dataset(root)

            rows = list(csv.DictReader(external_dataset_audit_csv(root).splitlines()))
            by_dataset = {row["dataset"]: row for row in rows}

            self.assertEqual(by_dataset["coco_public"]["json_files"], "1")
            self.assertEqual(by_dataset["coco_public"]["image_files"], "1")
            self.assertEqual(by_dataset["coco_public"]["source_annotations"], "2")
            self.assertEqual(by_dataset["coco_public"]["kept_annotations"], "2")
            self.assertEqual(by_dataset["coco_public"]["reviewable_annotations"], "2")

            self.assertEqual(by_dataset["image_json_public"]["source_annotations"], "4")
            self.assertEqual(by_dataset["image_json_public"]["kept_annotations"], "4")
            self.assertEqual(by_dataset["image_json_public"]["reviewable_annotations"], "0")
            self.assertEqual(by_dataset["image_json_public"]["not_reviewable_missing_image"], "4")
            self.assertEqual(by_dataset["image_json_public"]["missing_referenced_images"], "1")

            self.assertEqual(by_dataset["xml_public"]["xml_files"], "2")
            self.assertEqual(by_dataset["xml_public"]["source_annotations"], "4")
            self.assertEqual(by_dataset["xml_public"]["kept_annotations"], "2")
            self.assertEqual(by_dataset["xml_public"]["excluded_missing_image_file"], "2")

            self.assertEqual(by_dataset["crypto_yolo"]["yolo_label_txt_files"], "1")
            self.assertEqual(by_dataset["crypto_yolo"]["source_annotations"], "2")
            self.assertEqual(by_dataset["crypto_yolo"]["reviewable_annotations"], "2")
            self.assertEqual(by_dataset["crypto_yolo"]["missing_class_name_annotations"], "0")

    def test_external_dataset_audit_can_split_by_subfolder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_coco_dataset(root, dataset="split_coco", subfolder="train")
            self._write_coco_dataset(root, dataset="split_coco", subfolder="valid")
            self._write_yolo_data_yaml_dataset(root)

            rows = list(csv.DictReader(external_dataset_audit_csv(root, by_subfolder=True).splitlines()))
            by_group = {(row["dataset"], row["subfolder"]): row for row in rows}

            self.assertEqual(len(rows), 3)
            self.assertEqual(by_group[("split_coco", "train")]["source_annotations"], "2")
            self.assertEqual(by_group[("split_coco", "train")]["image_files"], "1")
            self.assertEqual(by_group[("split_coco", "valid")]["source_annotations"], "2")
            self.assertEqual(by_group[("split_coco", "valid")]["image_files"], "1")
            self.assertEqual(by_group[("crypto_yolo", "train")]["annotation_formats"], "yolo")
            self.assertEqual(by_group[("crypto_yolo", "train")]["source_annotation_files"], "1")
            self.assertEqual(by_group[("crypto_yolo", "train")]["source_annotations"], "2")
            self.assertEqual(by_group[("crypto_yolo", "train")]["reviewable_annotations"], "2")

    def test_external_dataset_audit_splits_coco_by_nested_image_subfolders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "nested_coco"
            self._write_jpeg(dataset_root / "train" / "images" / "a.jpg")
            self._write_jpeg(dataset_root / "valid" / "images" / "b.jpg")
            dataset_root.mkdir(exist_ok=True)
            (dataset_root / "labels.json").write_text(
                json.dumps(
                    {
                        "images": [
                            {"id": 1, "file_name": "train/images/a.jpg", "width": 10, "height": 10},
                            {"id": 2, "file_name": "valid/images/b.jpg", "width": 10, "height": 10},
                        ],
                        "categories": [{"id": 1, "name": "Ascaris lumbricoides"}],
                        "annotations": [
                            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 2, 3, 4]},
                            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [2, 3, 4, 5]},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            rows = list(csv.DictReader(external_dataset_audit_csv(root, by_subfolder=True).splitlines()))
            by_group = {(row["dataset"], row["subfolder"]): row for row in rows}

            self.assertEqual(by_group[("nested_coco", "train")]["source_annotations"], "1")
            self.assertEqual(by_group[("nested_coco", "train")]["image_files"], "1")
            self.assertEqual(by_group[("nested_coco", "train")]["reviewable_annotations"], "1")
            self.assertEqual(by_group[("nested_coco", "valid")]["source_annotations"], "1")
            self.assertEqual(by_group[("nested_coco", "valid")]["image_files"], "1")
            self.assertEqual(by_group[("nested_coco", "valid")]["reviewable_annotations"], "1")

    def test_external_dataset_audit_can_select_single_dataset_with_nested_subfolders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "AI4NTD-P15V2-P30Eval" / "ai4ntd-p3-0-ood" / "p3-0_2"
            self._write_jpeg(dataset_root / "images" / "example.jpg")
            (dataset_root / "labels.json").write_text(
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": "example.jpg", "width": 10, "height": 10}],
                        "categories": [{"id": 1, "name": "A. lumbricoides"}],
                        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 2, 3, 4]}],
                    }
                ),
                encoding="utf-8",
            )

            rows = list(
                csv.DictReader(
                    external_dataset_audit_csv(
                        root,
                        by_subfolder=True,
                        dataset="AI4NTD-P15V2-P30Eval",
                    ).splitlines()
                )
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["dataset"], "AI4NTD-P15V2-P30Eval")
            self.assertEqual(rows[0]["subfolder"], "ai4ntd-p3-0-ood/p3-0_2")
            self.assertEqual(rows[0]["image_files"], "1")
            self.assertEqual(rows[0]["source_annotations"], "1")
            self.assertEqual(rows[0]["reviewable_annotations"], "1")

    def test_counts_by_subfolder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_coco_dataset(root)
            self._write_csv_dataset(root)
            self._write_yolo_dataset(root)
            self._write_voc_xml_dataset(root)

            rows = list(csv.DictReader(external_counts_csv(root, by_subfolder=True).splitlines()))
            by_group = {(row["dataset"], row["subfolder"]): row for row in rows}

            self.assertEqual(by_group[("coco_public", ".")]["ascaris_lumbricoides"], "2")
            self.assertEqual(by_group[("csv_public", ".")]["taenia_spp"], "1")
            self.assertEqual(by_group[("xml_public", "Image_files")]["ascaris_lumbricoides"], "2")
            self.assertEqual(by_group[("yolo_public", "train/images")]["trichuris_trichiura"], "2")

    def test_counts_by_subfolder_does_not_reassign_coco_missing_images_by_basename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "AI4NTD-P15V2-P30Eval"
            animal_root = dataset_root / "ai4ntd-p3-0-ood" / "p3-0_1" / "animal"
            p3_2_root = dataset_root / "ai4ntd-p3-0-ood" / "p3-0_2"

            self._write_jpeg(p3_2_root / "images" / "same_name.jpg")
            self._write_coco_file(
                animal_root / "labels.json",
                image_name="same_name.jpg",
                boxes=[[1, 2, 3, 4], [2, 3, 4, 5]],
            )
            self._write_coco_file(
                p3_2_root / "labels.json",
                image_name="same_name.jpg",
                boxes=[[3, 4, 5, 6]],
            )

            rows = list(
                csv.DictReader(
                    external_counts_csv(
                        root,
                        by_subfolder=True,
                        dataset="AI4NTD-P15V2-P30Eval",
                    ).splitlines()
                )
            )
            by_group = {(row["dataset"], row["subfolder"]): row for row in rows}

            self.assertEqual(
                by_group[
                    ("AI4NTD-P15V2-P30Eval", "ai4ntd-p3-0-ood/p3-0_1/animal")
                ]["ascaris_lumbricoides"],
                "2",
            )
            self.assertEqual(
                by_group[
                    ("AI4NTD-P15V2-P30Eval", "ai4ntd-p3-0-ood/p3-0_2")
                ]["ascaris_lumbricoides"],
                "1",
            )
            self.assertNotIn(
                ("AI4NTD-P15V2-P30Eval", "ai4ntd-p3-0-ood/p3-0_2/images"),
                by_group,
            )

    def test_counts_skip_coco_zero_sized_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_coco_file(
                root / "coco_public" / "labels.json",
                image_name="example.jpg",
                boxes=[[1, 2, 3, 4], [312, 112.032, 8.064, 0]],
            )

            rows = list(csv.DictReader(external_counts_csv(root).splitlines()))

            self.assertEqual(rows[0]["ascaris_lumbricoides"], "1")

    def test_counts_sum_multiple_annotation_files_with_same_dataset_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_coco_dataset(root, dataset="split_coco", subfolder="train")
            self._write_coco_dataset(root, dataset="split_coco", subfolder="valid")

            rows = list(csv.DictReader(external_counts_csv(root).splitlines()))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["dataset"], "split_coco")
            self.assertEqual(rows[0]["ascaris_lumbricoides"], "4")

    def test_yolo_counts_use_data_yaml_class_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_yolo_data_yaml_dataset(root)

            rows = list(csv.DictReader(external_counts_csv(root).splitlines()))

            self.assertEqual(rows[0]["dataset"], "crypto_yolo")
            self.assertEqual(rows[0]["cryptosporidium"], "1")
            self.assertEqual(rows[0]["giardia"], "1")
            self.assertNotIn("class_0", rows[0])

    def test_review_bundle_preserves_labels_and_relative_image_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_coco_dataset(root)

            result = build_external_review_bundle(
                external_root=root,
                dataset_name="coco_public",
                output_dir=root / "review",
            )
            data = json.loads(result.proposals_path.read_text(encoding="utf-8"))

            self.assertEqual(result.total_images, 1)
            self.assertEqual(result.total_boxes, 2)
            self.assertEqual(data["images"][0]["file_name"], "images/example.jpg")
            self.assertEqual(data["annotations"][0]["label"], "ascaris_lumbricoides")

    def test_normalizes_common_label_variants(self) -> None:
        self.assertEqual(normalize_external_label("Ascaris-Lumbricoides"), "ascaris_lumbricoides")
        self.assertEqual(normalize_external_label("A. lumbricoides"), "ascaris_lumbricoides")
        self.assertEqual(normalize_external_label("ASC."), "ascaris_lumbricoides")
        self.assertEqual(normalize_external_label("C. parvum"), "cryptosporidium")
        self.assertEqual(normalize_external_label("Taenia spp. egg"), "taenia_spp")
        self.assertEqual(normalize_external_label("T. saginata"), "taenia_saginata")
        self.assertEqual(normalize_external_label("large egg (Fascilopsis buski)"), "fasciolopsis_buski")
        self.assertEqual(normalize_external_label("F. hepatica"), "fasciola_hepatica")
        self.assertEqual(normalize_external_label("s.mansoni"), "schistosoma_mansoni")
        self.assertEqual(normalize_external_label("SM"), "schistosoma_mansoni")
        self.assertEqual(normalize_external_label("SCH."), "schistosoma_mansoni")
        self.assertEqual(normalize_external_label("tt"), "trichuris_trichiura")
        self.assertEqual(normalize_external_label("TRI."), "trichuris_trichiura")
        self.assertEqual(normalize_external_label("T. trichuria"), "trichuris_trichiura")
        self.assertEqual(normalize_external_label("h"), "hookworm")
        self.assertEqual(normalize_external_label("HKW."), "hookworm")
        self.assertEqual(normalize_external_label("???"), "unknown")

    def test_infers_labels_from_external_image_filenames(self) -> None:
        self.assertEqual(label_from_image_filename("A. lumbricoides (10).jpeg"), "ascaris_lumbricoides")
        self.assertEqual(
            label_from_image_filename("A. lumbricoides (Fertilized egg) (10).JPG"),
            "ascaris_lumbricoides",
        )
        self.assertEqual(label_from_image_filename("Hookwarm (21).jpg"), "hookworm")
        self.assertEqual(label_from_image_filename("T. trichiura (19).jpg"), "trichuris_trichiura")
        self.assertEqual(
            label_from_image_filename("lab_microscope__trichuria_trichuris__881a531932c9.jpg"),
            "trichuria trichuris",
        )

    def test_label_normalization_report_records_observed_raw_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_coco_dataset(root, raw_label="A. lumbricoides")
            self._write_image_json_dataset(root)

            report = external_label_normalization_report(root)

            self.assertEqual(
                report["observed_labels"]["A. lumbricoides"]["normalized"],
                "ascaris_lumbricoides",
            )
            self.assertEqual(report["observed_labels"]["ASC."]["normalized"], "ascaris_lumbricoides")
            self.assertEqual(report["observed_labels"]["TRI."]["normalized"], "trichuris_trichiura")
            self.assertEqual(report["normalization_rules"]["a lumbricoides"], "ascaris_lumbricoides")
            self.assertIn("ASC.", report["canonical_labels"]["ascaris_lumbricoides"])

    def test_discovers_cysts_coco_subdatasets_as_separate_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "cysts_dataset_all" / "brightfield_sample"
            dataset_root.mkdir(parents=True)
            (dataset_root / "brightfield_sample_coco_annos.json").write_text(
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": "BS0000.jpg", "width": 10, "height": 10}],
                        "categories": [{"id": 0, "name": "Crypto"}],
                        "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [1, 2, 3, 4]}],
                    }
                ),
                encoding="utf-8",
            )

            datasets = discover_external_datasets(root)

            self.assertEqual([dataset.name for dataset in datasets], ["cysts_dataset_all/brightfield_sample"])

    def test_converts_pascal_voc_xml_to_single_coco_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_voc_xml_dataset(root)

            result = convert_external_xml_annotations(root, "xml_public")
            data = json.loads(result.json_path.read_text(encoding="utf-8"))

            self.assertEqual(result.xml_files, 1)
            self.assertEqual(result.images, 1)
            self.assertEqual(result.annotations, 2)
            self.assertEqual(result.json_path, (root / "xml_public" / "annotations.json").resolve())
            self.assertEqual(data["images"][0]["file_name"], "Image_files/example.jpg")
            self.assertEqual(data["categories"], [{"id": 1, "name": "ascaris_lumbricoides"}])
            self.assertEqual(data["annotations"][0]["bbox"], [1.0, 2.0, 4.0, 6.0])

    def test_xml_dataset_is_in_external_counts_and_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_voc_xml_dataset(root)

            rows = list(csv.DictReader(external_counts_csv(root).splitlines()))
            result = build_external_review_bundle(root, "xml_public", root / "review")

            self.assertEqual(rows[0]["dataset"], "xml_public")
            self.assertEqual(rows[0]["ascaris_lumbricoides"], "2")
            self.assertEqual(result.total_images, 1)
            self.assertEqual(result.total_boxes, 2)

    def test_external_counts_skips_xml_with_missing_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_voc_xml_dataset(root, stem="valid")
            self._write_voc_xml_dataset(
                root,
                stem="missing",
                filename="does_not_exist.jpg",
                write_image=False,
            )

            rows = list(csv.DictReader(external_counts_csv(root).splitlines()))
            subfolder_rows = list(csv.DictReader(external_counts_csv(root, by_subfolder=True).splitlines()))

            self.assertEqual(rows[0]["dataset"], "xml_public")
            self.assertEqual(rows[0]["ascaris_lumbricoides"], "2")
            self.assertEqual(len(subfolder_rows), 1)
            self.assertEqual(subfolder_rows[0]["subfolder"], "Image_files")
            self.assertEqual(subfolder_rows[0]["ascaris_lumbricoides"], "2")

    def test_xml_conversion_skips_missing_referenced_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_voc_xml_dataset(root, stem="valid")
            self._write_voc_xml_dataset(
                root,
                stem="missing",
                filename="does_not_exist.jpg",
                write_image=False,
            )

            result = convert_external_xml_annotations(root, "xml_public")
            data = json.loads(result.json_path.read_text(encoding="utf-8"))

            self.assertEqual(result.xml_files, 2)
            self.assertEqual(result.missing_images, 1)
            self.assertEqual(result.images, 1)
            self.assertEqual(result.annotations, 2)
            self.assertEqual(data["images"][0]["file_name"], "Image_files/valid.jpg")

    def test_xml_conversion_uses_same_stem_image_when_filename_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_voc_xml_dataset(root, stem="stale", filename="does_not_exist.jpg")

            result = convert_external_xml_annotations(root, "xml_public")
            data = json.loads(result.json_path.read_text(encoding="utf-8"))

            self.assertEqual(result.missing_images, 0)
            self.assertEqual(data["images"][0]["file_name"], "Image_files/stale.jpg")

    def test_xml_conversion_ignores_unlabeled_test_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_voc_xml_dataset(root)
            self._write_voc_xml_dataset(root, folder="Test_samples", write_image=False)

            result = convert_external_xml_annotations(root, "xml_public")
            data = json.loads(result.json_path.read_text(encoding="utf-8"))

            self.assertEqual(result.xml_files, 1)
            self.assertEqual(result.images, 1)
            self.assertEqual(result.annotations, 2)
            self.assertEqual(data["images"][0]["file_name"], "Image_files/example.jpg")

    def _write_coco_dataset(
        self,
        root: Path,
        dataset: str = "coco_public",
        subfolder: str | None = None,
        raw_label: str = "Ascaris lumbricoides",
    ) -> None:
        dataset_root = root / dataset
        annotation_dir = dataset_root / subfolder if subfolder else dataset_root
        image_dir = annotation_dir / "images"
        image_dir.mkdir(parents=True)
        self._write_jpeg(image_dir / "example.jpg")
        file_name = "images/example.jpg" if subfolder else "example.jpg"
        (annotation_dir / "labels.json").write_text(
            json.dumps(
                {
                    "images": [{"id": 1, "file_name": file_name, "width": 10, "height": 10}],
                    "categories": [{"id": 1, "name": raw_label}],
                    "annotations": [
                        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 2, 3, 4]},
                        {"id": 2, "image_id": 1, "category_id": 1, "bbox": [2, 3, 4, 5]},
                    ],
                }
            ),
            encoding="utf-8",
        )

    def _write_image_json_dataset(self, root: Path) -> None:
        annotations_dir = root / "image_json_public" / "annotations"
        annotations_dir.mkdir(parents=True)
        (annotations_dir / "image_00001.json").write_text(
            json.dumps(
                {
                    "filename": "image_00001.png",
                    "size": {"width": 100, "height": 80},
                    "objects": [
                        {"label": "ASC.", "bbox": [1, 2, 11, 12], "area": 100},
                        {"label": "TRI.", "bbox": [3, 4, 13, 14], "area": 100},
                        {"label": "HKW.", "bbox": [5, 6, 15, 16], "area": 100},
                        {"label": "SCH.", "bbox": [7, 8, 17, 18], "area": 100},
                    ],
                }
            ),
            encoding="utf-8",
        )

    def _write_csv_dataset(self, root: Path) -> None:
        dataset_root = root / "csv_public"
        dataset_root.mkdir()
        (dataset_root / "model_class.csv").write_text("0,Taenia spp_\n", encoding="utf-8")
        (dataset_root / "model_label.csv").write_text(
            ",img_name,class_id,xmin,xmax,ymin,ymax\n0,taenia.jpg,0,1,5,2,8\n",
            encoding="utf-8",
        )

    def _write_yolo_dataset(self, root: Path) -> None:
        labels_dir = root / "yolo_public" / "train" / "labels"
        labels_dir.mkdir(parents=True)
        (labels_dir / "Trichuris-Trichiura--1-_jpg.rf.abc.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n0 0.4 0.4 0.1 0.1\n",
            encoding="utf-8",
        )

    def _write_yolo_data_yaml_dataset(self, root: Path) -> None:
        dataset_root = root / "crypto_yolo"
        labels_dir = dataset_root / "train" / "labels"
        images_dir = dataset_root / "train" / "images"
        labels_dir.mkdir(parents=True)
        images_dir.mkdir(parents=True)
        self._write_jpeg(images_dir / "example.jpg")
        (dataset_root / "data.yaml").write_text(
            "names:\n- Giardia\n- C. parvum\nnc: 2\ntrain: train/images\nval: valid/images\n",
            encoding="utf-8",
        )
        (labels_dir / "example.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.1 0.1\n",
            encoding="utf-8",
        )

    def _write_voc_xml_dataset(
        self,
        root: Path,
        folder: str = "Image_files",
        stem: str = "example",
        filename: str | None = None,
        write_image: bool = True,
    ) -> None:
        image_dir = root / "xml_public" / folder
        image_dir.mkdir(parents=True, exist_ok=True)
        if write_image:
            self._write_jpeg(image_dir / f"{stem}.jpg")
        xml_filename = filename or f"{stem}.jpg"
        (image_dir / f"{stem}.xml").write_text(
            f"""
<annotation>
  <filename>{xml_filename}</filename>
  <size><width>10</width><height>10</height><depth>3</depth></size>
  <object>
    <name>al</name>
    <bndbox><xmin>1</xmin><ymin>2</ymin><xmax>5</xmax><ymax>8</ymax></bndbox>
  </object>
  <object>
    <name>al</name>
    <bndbox><xmin>2</xmin><ymin>3</ymin><xmax>6</xmax><ymax>9</ymax></bndbox>
  </object>
</annotation>
""".strip(),
            encoding="utf-8",
        )

    def _write_coco_file(self, path: Path, image_name: str, boxes: list[list[float]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "images": [{"id": 1, "file_name": image_name, "width": 10, "height": 10}],
                    "categories": [{"id": 1, "name": "Ascaris lumbricoides"}],
                    "annotations": [
                        {"id": index + 1, "image_id": 1, "category_id": 1, "bbox": box}
                        for index, box in enumerate(boxes)
                    ],
                }
            ),
            encoding="utf-8",
        )

    def _write_jpeg(self, path: Path) -> None:
        from PIL import Image

        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (10, 10), "white")
        image.save(path)


if __name__ == "__main__":
    unittest.main()
