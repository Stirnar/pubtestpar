from __future__ import annotations

import csv
import io
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
_IMAGE_INDEX_CACHE: dict[Path, dict[str, list[Path]]] = {}
_YOLO_CLASS_NAMES_CACHE: dict[Path, dict[int, str]] = {}
_YOLO_DATA_YAML_PATHS_CACHE: dict[Path, list[Path]] = {}
_YOLO_LABEL_CLASS_NAMES_CACHE: dict[tuple[Path, Path], dict[int, str]] = {}
_IGNORED_EXTERNAL_DIR_NAMES = {"test_samples"}

_LABEL_FIXES = {
    "a lumbricoides": "ascaris_lumbricoides",
    "asc": "ascaris_lumbricoides",
    "ascaris lumbricoides": "ascaris_lumbricoides",
    "ancylostoma spp": "ancylostoma_spp",
    "al": "ascaris_lumbricoides",
    "blastocystis spp": "blastocystis_spp",
    "c philippinensis": "capillaria_philippinensis",
    "capillaria philippinensis": "capillaria_philippinensis",
    "chillomastix": "chilomastix",
    "c parvum": "cryptosporidium",
    "crypto": "cryptosporidium",
    "cryptosporidium parvum": "cryptosporidium",
    "cryptosporidium": "cryptosporidium",
    "d caninum": "dipylidium_caninum",
    "d latum": "diphyllobothrium_latum",
    "e coli": "entamoeba_coli",
    "e histolytica": "entamoeba_histolytica",
    "e nana": "endolimax_nana",
    "e pancreaticum": "eurytrema_pancreaticum",
    "echinostoma spp": "echinostoma_spp",
    "endolimax nana": "endolimax_nana",
    "enterobius vermicularis": "enterobius_vermicularis",
    "entamoeba": "entamoeba",
    "entamoeba coli": "entamoeba_coli",
    "entamoeba hart": "entamoeba_hart",
    "entamoeba histolytica": "entamoeba_histolytica",
    "eurytrema pancreaticum": "eurytrema_pancreaticum",
    "f buski": "fasciolopsis_buski",
    "f hepatica": "fasciola_hepatica",
    "fasciola hepatica": "fasciola_hepatica",
    "fasciola spp": "fasciola_spp",
    "fasciolopsis buski": "fasciolopsis_buski",
    "fascilopsis buski": "fasciolopsis_buski",
    "giardia": "giardia",
    "g duodenalis": "giardia_duodenalis",
    "g hominis": "giardia_hominis",
    "giardia duodenalis": "giardia_duodenalis",
    "giardia hominis": "giardia_hominis",
    "h": "hookworm",
    "h diminuta": "hymenolepis_diminuta",
    "h nana": "hymenolepis_nana",
    "haplorchis spp": "haplorchis_spp",
    "hkw": "hookworm",
    "hookwarm": "hookworm",
    "hookworm": "hookworm",
    "hw": "hookworm",
    "hymenolepis": "hymenolepis",
    "hymenolepis diminuta": "hymenolepis_diminuta",
    "hymenolepis nana": "hymenolepis_nana",
    "i butschlii": "iodamoeba_butschlii",
    "iodamoeba butschlii": "iodamoeba_butschlii",
    "o viverrini": "opisthorchis_viverrini",
    "opisthorchis viverrine": "opisthorchis_viverrini",
    "opisthorchis viverrini": "opisthorchis_viverrini",
    "paragonimus spp": "paragonimus_spp",
    "s haematobium": "schistosoma_haematobium",
    "s japonicum": "schistosoma_japonicum",
    "s mansoni": "schistosoma_mansoni",
    "s mekongi": "schistosoma_mekongi",
    "s stercoralis": "strongyloides_stercoralis",
    "sch": "schistosoma_mansoni",
    "schistosoma haematobium": "schistosoma_haematobium",
    "schistosoma japonicum": "schistosoma_japonicum",
    "schistosoma mansoni": "schistosoma_mansoni",
    "schistosoma mekongi": "schistosoma_mekongi",
    "schistosoma": "schistosoma",
    "sm": "schistosoma_mansoni",
    "spirometra spp": "spirometra_spp",
    "strongyloides stercoralis": "strongyloides_stercoralis",
    "t orientalis": "trichostrongylus_orientalis",
    "t saginata": "taenia_saginata",
    "t trichiura": "trichuris_trichiura",
    "t trichuria": "trichuris_trichiura",
    "t trichuris": "trichuris_trichiura",
    "taenia": "taenia_spp",
    "taenia saginata": "taenia_saginata",
    "taenia sp": "taenia_spp",
    "taenia spp": "taenia_spp",
    "toxocara spp": "toxocara_spp",
    "tricocefalo": "trichuris_trichiura",
    "tri": "trichuris_trichiura",
    "trichostrongylus orientalis": "trichostrongylus_orientalis",
    "trichuria": "trichuris_trichiura",
    "trichuria trichuris": "trichuris_trichiura",
    "trichuris": "trichuris_trichiura",
    "trichuris trichiura": "trichuris_trichiura",
    "tt": "trichuris_trichiura",
    "unknown": "unknown",
}


@dataclass(frozen=True)
class ExternalAnnotation:
    dataset: str
    image_name: str
    width: int
    height: int
    label: str
    bbox: tuple[float, float, float, float]
    source_format: str
    source_path: Path


@dataclass(frozen=True)
class ExternalDataset:
    name: str
    root: Path
    annotations: tuple[ExternalAnnotation, ...]


@dataclass(frozen=True)
class ReviewBundleResult:
    dataset: str
    images_dir: Path
    proposals_path: Path
    annotations_path: Path
    total_images: int
    total_boxes: int
    missing_images: int
    skipped_boxes: int


@dataclass(frozen=True)
class XmlConversionResult:
    dataset: str
    json_path: Path
    xml_files: int
    images: int
    annotations: int
    missing_images: int


def external_counts_csv(
    external_root: Path,
    by_subfolder: bool = False,
    dataset: str | None = None,
) -> str:
    root = external_root.expanduser().resolve()
    datasets = [load_external_dataset(root, dataset)] if dataset else discover_external_datasets(root)
    if by_subfolder:
        return _external_counts_by_subfolder_csv(datasets)

    counts_by_dataset = {}
    for dataset in datasets:
        counts_by_dataset.setdefault(dataset.name, Counter()).update(
            annotation.label for annotation in dataset.annotations
        )
    labels = sorted({label for counts in counts_by_dataset.values() for label in counts})

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["dataset", *labels])
    for dataset_name in sorted(counts_by_dataset):
        counts = counts_by_dataset[dataset_name]
        writer.writerow([dataset_name, *[counts.get(label, 0) for label in labels]])
    return output.getvalue()


def external_label_normalization_json(external_root: Path) -> str:
    return json.dumps(external_label_normalization_report(external_root), indent=2, sort_keys=True) + "\n"


def external_label_normalization_report(external_root: Path) -> dict[str, Any]:
    root = external_root.expanduser().resolve()
    observed = _collect_raw_external_labels(root)
    observed_labels = {}
    canonical_labels: dict[str, list[str]] = {}
    for raw_label in sorted(observed, key=lambda value: value.lower()):
        normalized = normalize_external_label(raw_label)
        sources = sorted(observed[raw_label])
        observed_labels[raw_label] = {
            "normalized": normalized,
            "source_count": len(sources),
            "sources": sources[:25],
        }
        canonical_labels.setdefault(normalized, []).append(raw_label)

    return {
        "canonical_labels": dict(sorted(canonical_labels.items())),
        "normalization_rules": dict(sorted(_LABEL_FIXES.items())),
        "observed_labels": observed_labels,
    }


def _external_counts_by_subfolder_csv(datasets: Sequence[ExternalDataset]) -> str:
    counts_by_group = {}
    for dataset in datasets:
        for annotation in dataset.annotations:
            key = (dataset.name, _annotation_subfolder(annotation, dataset.root))
            counts_by_group.setdefault(key, Counter())[annotation.label] += 1

    labels = sorted({label for counts in counts_by_group.values() for label in counts})

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["dataset", "subfolder", *labels])
    for dataset_name, subfolder in sorted(counts_by_group):
        counts = counts_by_group[(dataset_name, subfolder)]
        writer.writerow([dataset_name, subfolder, *[counts.get(label, 0) for label in labels]])
    return output.getvalue()


def discover_external_datasets(external_root: Path) -> list[ExternalDataset]:
    root = external_root.expanduser().resolve()
    datasets: list[ExternalDataset] = []
    datasets.extend(_discover_coco_datasets(root))
    datasets.extend(_discover_image_json_datasets(root, excluded_roots={dataset.root for dataset in datasets}))
    datasets.extend(_discover_usb_csv_datasets(root))
    datasets.extend(_discover_voc_xml_datasets(root, excluded_roots={dataset.root for dataset in datasets}))
    datasets.extend(_discover_yolo_datasets(root, excluded_roots={dataset.root for dataset in datasets}))
    return sorted(datasets, key=lambda dataset: dataset.name)


def load_external_dataset(external_root: Path, dataset_name: str) -> ExternalDataset:
    external_root = external_root.expanduser().resolve()
    dataset_root = (external_root / dataset_name).resolve()
    if not dataset_root.exists():
        available = ", ".join(
            sorted(path.relative_to(external_root).as_posix() for path in external_root.iterdir() if path.is_dir())
        )
        raise ValueError(f"Unknown external dataset {dataset_name!r}. Available top-level datasets: {available}")

    dataset = _discover_dataset_at_root(dataset_name, dataset_root)
    if dataset is None:
        raise ValueError(f"No supported annotations found for external dataset {dataset_name!r}.")
    return dataset


def convert_external_xml_annotations(
    external_root: Path,
    dataset_name: str,
    output: Path | None = None,
) -> XmlConversionResult:
    external_root = external_root.expanduser().resolve()
    dataset_root = external_root / dataset_name
    if not dataset_root.exists():
        raise ValueError(f"External dataset not found: {dataset_root}")
    document, xml_files, missing_images = _voc_xml_coco_document(dataset_root)
    output_path = output.expanduser().resolve() if output else dataset_root / "annotations.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return XmlConversionResult(
        dataset=dataset_name,
        json_path=output_path,
        xml_files=xml_files,
        images=len(document["images"]),
        annotations=len(document["annotations"]),
        missing_images=missing_images,
    )


def build_external_review_bundle(
    external_root: Path,
    dataset_name: str,
    output_dir: Path,
    images_dir: Path | None = None,
    annotations_path: Path | None = None,
) -> ReviewBundleResult:
    dataset = load_external_dataset(external_root, dataset_name)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    review_images_dir = images_dir.expanduser().resolve() if images_dir else dataset.root
    annotations_path = (
        annotations_path.expanduser().resolve()
        if annotations_path
        else output_dir / "annotations.json"
    )

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    image_ids: dict[str, int] = {}
    missing_images: set[str] = set()
    skipped_boxes = 0

    for annotation in dataset.annotations:
        review_name, exists = _review_image_name(
            annotation.image_name,
            dataset.root,
            review_images_dir,
        )
        if not exists:
            missing_images.add(review_name)

        if not _valid_review_box(annotation):
            skipped_boxes += 1
            continue

        image_id = image_ids.get(review_name)
        if image_id is None:
            image_id = len(image_ids) + 1
            image_ids[review_name] = image_id
            images.append(
                {
                    "id": image_id,
                    "file_name": review_name,
                    "width": annotation.width,
                    "height": annotation.height,
                }
            )

        x, y, width, height = annotation.bbox
        annotations.append(
            {
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x, y, width, height],
                "area": width * height,
                "iscrowd": 0,
                "score": 1.0,
                "label": annotation.label,
                "detection_method": f"external_{annotation.source_format}",
            }
        )

    proposals_path = output_dir / "proposals.json"
    proposals_path.write_text(
        json.dumps(
            {
                "info": {
                    "description": f"External annotation review bundle for {dataset.name}",
                    "generator": "external_datasets.py",
                },
                "images": images,
                "annotations": annotations,
                "categories": [{"id": 1, "name": "external_annotation"}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return ReviewBundleResult(
        dataset=dataset.name,
        images_dir=review_images_dir,
        proposals_path=proposals_path,
        annotations_path=annotations_path,
        total_images=len(images),
        total_boxes=len(annotations),
        missing_images=len(missing_images),
        skipped_boxes=skipped_boxes,
    )


def summarize_review_bundle(result: ReviewBundleResult) -> str:
    lines = [
        f"Dataset: {result.dataset}",
        f"Images dir: {result.images_dir}",
        f"Proposals: {result.proposals_path}",
        f"Annotations output: {result.annotations_path}",
        f"Images in review: {result.total_images}",
        f"Boxes in review: {result.total_boxes}",
    ]
    if result.missing_images:
        lines.append(f"Missing local image files: {result.missing_images}")
    if result.skipped_boxes:
        lines.append(f"Skipped boxes without pixel dimensions: {result.skipped_boxes}")
    return "\n".join(lines)


def summarize_xml_conversion(result: XmlConversionResult) -> str:
    return "\n".join(
        [
            f"Dataset: {result.dataset}",
            f"XML files: {result.xml_files}",
            f"Images: {result.images}",
            f"Annotations: {result.annotations}",
            f"Missing images: {result.missing_images}",
            f"JSON: {result.json_path}",
        ]
    )


def _discover_coco_datasets(external_root: Path) -> list[ExternalDataset]:
    datasets: list[ExternalDataset] = []
    for path in sorted(external_root.rglob("*.json")):
        if _is_ignored_external_path(path):
            continue
        if not _should_use_json_annotation(path):
            continue
        if not _is_coco(_read_json(path)):
            continue

        dataset_name = _dataset_name_for_annotation(path, external_root)
        dataset_root = _dataset_root_for_annotation(path, external_root)
        datasets.append(ExternalDataset(dataset_name, dataset_root, tuple(_read_coco_annotations(path, dataset_name))))
    return datasets


def _discover_image_json_datasets(external_root: Path, excluded_roots: set[Path]) -> list[ExternalDataset]:
    datasets: list[ExternalDataset] = []
    top_level_roots = [path for path in external_root.iterdir() if path.is_dir()]
    for dataset_root in sorted(top_level_roots):
        resolved_root = dataset_root.resolve()
        if any(_is_relative_to(resolved_root, excluded.resolve()) for excluded in excluded_roots):
            continue

        annotations: list[ExternalAnnotation] = []
        for path in _annotation_files(dataset_root, "*.json"):
            try:
                data = _read_json(path)
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                continue
            if _is_image_json_annotation(data):
                annotations.extend(_read_image_json_annotations(path, dataset_root, dataset_root.name, data))

        if annotations:
            datasets.append(ExternalDataset(dataset_root.name, dataset_root, tuple(annotations)))
    return datasets


def _discover_usb_csv_datasets(external_root: Path) -> list[ExternalDataset]:
    datasets: list[ExternalDataset] = []
    for labels_path in sorted(external_root.rglob("model_label.csv")):
        classes_path = labels_path.with_name("model_class.csv")
        if not classes_path.exists():
            continue
        dataset_name = _dataset_name_for_annotation(labels_path, external_root)
        dataset_root = _dataset_root_for_annotation(labels_path, external_root)
        datasets.append(
            ExternalDataset(
                dataset_name,
                dataset_root,
                tuple(_read_usb_csv_annotations(labels_path, classes_path, dataset_name, dataset_root)),
            )
        )
    return datasets


def _discover_voc_xml_datasets(external_root: Path, excluded_roots: set[Path]) -> list[ExternalDataset]:
    datasets: list[ExternalDataset] = []
    top_level_roots = [path for path in external_root.iterdir() if path.is_dir()]
    for dataset_root in sorted(top_level_roots):
        resolved_root = dataset_root.resolve()
        if any(_is_relative_to(resolved_root, excluded.resolve()) for excluded in excluded_roots):
            continue
        xml_paths = _annotation_files(dataset_root, "*.xml")
        if not xml_paths:
            continue
        annotations: list[ExternalAnnotation] = []
        for xml_path in xml_paths:
            annotations.extend(_read_voc_xml_annotations(xml_path, dataset_root, dataset_root.name))
        datasets.append(ExternalDataset(dataset_root.name, dataset_root, tuple(annotations)))
    return datasets


def _discover_yolo_datasets(external_root: Path, excluded_roots: set[Path]) -> list[ExternalDataset]:
    datasets: list[ExternalDataset] = []
    top_level_roots = [path for path in external_root.iterdir() if path.is_dir()]
    for dataset_root in sorted(top_level_roots):
        resolved_root = dataset_root.resolve()
        if any(_is_relative_to(resolved_root, excluded.resolve()) for excluded in excluded_roots):
            continue
        label_paths = _annotation_files(dataset_root, "labels/*.txt")
        if not label_paths:
            continue

        dataset_name = dataset_root.name
        datasets.append(ExternalDataset(dataset_name, dataset_root, tuple(_read_yolo_annotations(dataset_root, dataset_name))))
    return datasets


def _discover_dataset_at_root(dataset_name: str, dataset_root: Path) -> ExternalDataset | None:
    coco_annotations: list[ExternalAnnotation] = []
    for path in sorted(dataset_root.rglob("*.json")):
        if not _should_use_json_annotation(path):
            continue
        if _is_coco(_read_json(path)):
            coco_annotations.extend(_read_coco_annotations(path, dataset_name))
    if coco_annotations:
        return ExternalDataset(dataset_name, dataset_root, tuple(coco_annotations))

    image_json_annotations: list[ExternalAnnotation] = []
    for path in _annotation_files(dataset_root, "*.json"):
        try:
            data = _read_json(path)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue
        if _is_image_json_annotation(data):
            image_json_annotations.extend(_read_image_json_annotations(path, dataset_root, dataset_name, data))
    if image_json_annotations:
        return ExternalDataset(dataset_name, dataset_root, tuple(image_json_annotations))

    for labels_path in sorted(dataset_root.rglob("model_label.csv")):
        classes_path = labels_path.with_name("model_class.csv")
        if classes_path.exists():
            return ExternalDataset(
                dataset_name,
                dataset_root,
                tuple(_read_usb_csv_annotations(labels_path, classes_path, dataset_name, dataset_root)),
            )

    xml_paths = _annotation_files(dataset_root, "*.xml")
    if xml_paths:
        annotations: list[ExternalAnnotation] = []
        for xml_path in xml_paths:
            annotations.extend(_read_voc_xml_annotations(xml_path, dataset_root, dataset_name))
        return ExternalDataset(dataset_name, dataset_root, tuple(annotations))

    if _annotation_files(dataset_root, "labels/*.txt"):
        return ExternalDataset(dataset_name, dataset_root, tuple(_read_yolo_annotations(dataset_root, dataset_name)))

    return None


def _read_coco_annotations(path: Path, dataset_name: str) -> list[ExternalAnnotation]:
    data = _read_json(path)
    categories = _coco_categories(data)
    images = _coco_images(data)
    annotations: list[ExternalAnnotation] = []
    for row in data.get("annotations", []):
        if not isinstance(row, Mapping):
            continue
        image = images.get(row.get("image_id"))
        if not image:
            continue
        bbox = _coerce_bbox(row.get("bbox"))
        if not _valid_bbox(bbox):
            continue
        label = categories.get(row.get("category_id"), f"class_{row.get('category_id')}")
        annotations.append(
            ExternalAnnotation(
                dataset=dataset_name,
                image_name=_image_name_from_coco(image),
                width=int(image.get("width") or 0),
                height=int(image.get("height") or 0),
                label=normalize_external_label(label),
                bbox=bbox,
                source_format="coco",
                source_path=path,
            )
        )
    return annotations


def _read_usb_csv_annotations(
    labels_path: Path,
    classes_path: Path,
    dataset_name: str,
    dataset_root: Path,
) -> list[ExternalAnnotation]:
    classes = _read_model_class_csv(classes_path)
    annotations: list[ExternalAnnotation] = []
    with labels_path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            label = classes.get(row.get("class_id", ""), f"class_{row.get('class_id', '')}")
            image_name = str(row.get("img_name") or "")
            bbox = _bbox_from_minmax(row)
            if not image_name or not bbox:
                continue
            width, height = _image_size_for_name(image_name, dataset_root)
            annotations.append(
                ExternalAnnotation(
                    dataset=dataset_name,
                    image_name=image_name,
                    width=width,
                    height=height,
                    label=normalize_external_label(label),
                    bbox=bbox,
                    source_format="csv",
                    source_path=labels_path,
                )
            )
    return annotations


def _read_image_json_annotations(
    path: Path,
    dataset_root: Path,
    dataset_name: str,
    data: Mapping[str, Any] | None = None,
) -> list[ExternalAnnotation]:
    data = data or _read_json(path)
    if not _is_image_json_annotation(data):
        return []

    image_name = str(data.get("filename") or path.with_suffix(".jpg").name).replace("\\", "/")
    width, height = _image_json_size(data.get("size"))
    if width <= 0 or height <= 0:
        width, height = _image_size_for_name(image_name, dataset_root)

    annotations: list[ExternalAnnotation] = []
    for obj in data.get("objects", []):
        if not isinstance(obj, Mapping):
            continue
        raw_label = str(obj.get("label") or obj.get("name") or obj.get("category") or "unknown")
        bbox = _bbox_from_image_json_object(obj)
        if bbox is None:
            continue
        annotations.append(
            ExternalAnnotation(
                dataset=dataset_name,
                image_name=image_name,
                width=width,
                height=height,
                label=normalize_external_label(raw_label),
                bbox=bbox,
                source_format="image_json",
                source_path=path,
            )
        )
    return annotations


def _read_yolo_annotations(dataset_root: Path, dataset_name: str) -> list[ExternalAnnotation]:
    annotations: list[ExternalAnnotation] = []
    label_paths = _annotation_files(dataset_root, "labels/*.txt")
    for label_path in label_paths:
        image_name = _image_name_for_yolo_label(label_path, dataset_root)
        label = _label_from_yolo_filename(label_path)
        class_names = _yolo_class_names_for_label_path(label_path, dataset_root)
        width, height = _image_size_for_name(image_name, dataset_root)
        for class_id, x_center, y_center, box_width, box_height in _read_yolo_rows(label_path):
            row_label = label or class_names.get(class_id) or f"class_{class_id}"
            bbox = _yolo_bbox_to_xywh(x_center, y_center, box_width, box_height, width, height)
            annotations.append(
                ExternalAnnotation(
                    dataset=dataset_name,
                    image_name=image_name,
                    width=width,
                    height=height,
                    label=normalize_external_label(row_label),
                    bbox=bbox,
                    source_format="yolo",
                    source_path=label_path,
                )
            )
    return annotations


def _voc_xml_coco_document(dataset_root: Path) -> tuple[dict[str, Any], int, int]:
    xml_paths = _annotation_files(dataset_root, "*.xml")
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    category_to_id: dict[str, int] = {}
    image_to_id: dict[str, int] = {}
    missing_images = 0

    for xml_path in xml_paths:
        if _xml_referenced_image_path(xml_path, dataset_root) is None:
            missing_images += 1
            continue
        for annotation in _read_voc_xml_annotations(xml_path, dataset_root, dataset_root.name):
            image_id = image_to_id.get(annotation.image_name)
            if image_id is None:
                image_id = len(image_to_id) + 1
                image_to_id[annotation.image_name] = image_id
                images.append(
                    {
                        "id": image_id,
                        "file_name": annotation.image_name,
                        "width": annotation.width,
                        "height": annotation.height,
                    }
                )

            category_id = category_to_id.get(annotation.label)
            if category_id is None:
                category_id = len(category_to_id) + 1
                category_to_id[annotation.label] = category_id
            x, y, width, height = annotation.bbox
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x, y, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )

    categories = [{"id": id_, "name": name} for name, id_ in sorted(category_to_id.items(), key=lambda item: item[1])]
    return (
        {
            "info": {
                "description": f"Converted Pascal VOC XML annotations for {dataset_root.name}",
                "generator": "external_datasets.py",
            },
            "images": images,
            "annotations": annotations,
            "categories": categories,
        },
        len(xml_paths),
        missing_images,
    )


def _read_voc_xml_annotations(xml_path: Path, dataset_root: Path, dataset_name: str) -> list[ExternalAnnotation]:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return []

    filename = _xml_text(root, "filename") or xml_path.with_suffix(".jpg").name
    image_path = _xml_image_path(filename, xml_path, dataset_root)
    if image_path is None:
        return []
    image_name = _relative_or_name(image_path, dataset_root)
    width = _xml_int(root, "size/width")
    height = _xml_int(root, "size/height")
    if width <= 0 or height <= 0:
        width, height = _image_size_for_name(image_name, dataset_root)

    annotations: list[ExternalAnnotation] = []
    for obj in root.findall("object"):
        label = normalize_external_label(_xml_text(obj, "name") or "parasite")
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = _xml_float(box, "xmin")
        ymin = _xml_float(box, "ymin")
        xmax = _xml_float(box, "xmax")
        ymax = _xml_float(box, "ymax")
        bbox = (xmin, ymin, xmax - xmin, ymax - ymin)
        annotations.append(
            ExternalAnnotation(
                dataset=dataset_name,
                image_name=image_name,
                width=width,
                height=height,
                label=label,
                bbox=bbox,
                source_format="voc_xml",
                source_path=xml_path,
            )
        )
    return annotations


def _annotation_files(root: Path, pattern: str) -> list[Path]:
    return sorted(path for path in root.rglob(pattern) if path.is_file() and not _is_ignored_external_path(path))


def _is_ignored_external_path(path: Path) -> bool:
    return any(part.lower() in _IGNORED_EXTERNAL_DIR_NAMES for part in path.parts)


def _xml_referenced_image_path(xml_path: Path, dataset_root: Path) -> Path | None:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None
    filename = _xml_text(root, "filename") or xml_path.with_suffix(".jpg").name
    return _xml_image_path(filename, xml_path, dataset_root)


def _xml_image_path(filename: str, xml_path: Path, dataset_root: Path) -> Path | None:
    value = filename.replace("\\", "/")
    image_name = Path(value).name
    sibling = xml_path.with_name(image_name)
    if sibling.exists() and sibling.suffix.lower() in IMAGE_SUFFIXES:
        return sibling

    found = _find_image_path(value, dataset_root) or _find_image_path(image_name, dataset_root)
    if found:
        return found

    for suffix in IMAGE_SUFFIXES:
        candidate = xml_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def _xml_text(root: ET.Element, path: str) -> str | None:
    element = root.find(path)
    if element is None or element.text is None:
        return None
    return element.text.strip()


def _xml_int(root: ET.Element, path: str) -> int:
    try:
        return int(float(_xml_text(root, path) or "0"))
    except ValueError:
        return 0


def _xml_float(root: ET.Element, path: str) -> float:
    try:
        return float(_xml_text(root, path) or "0")
    except ValueError:
        return 0.0


def _should_use_json_annotation(path: Path) -> bool:
    if path.name.endswith("_coco_annos.json"):
        return True

    is_candidate = (
        path.name in {"annotations.json", "labels.json"}
        or path.stem.lower() == path.parent.name.lower()
    )
    if not is_candidate:
        return False

    return not any(sibling.name.endswith("_coco_annos.json") for sibling in path.parent.glob("*.json"))


def _is_coco(data: Any) -> bool:
    return (
        isinstance(data, Mapping)
        and isinstance(data.get("images"), list)
        and isinstance(data.get("annotations"), list)
        and isinstance(data.get("categories"), list)
    )


def _is_image_json_annotation(data: Any) -> bool:
    return (
        isinstance(data, Mapping)
        and isinstance(data.get("filename"), str)
        and isinstance(data.get("objects"), list)
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _coco_categories(data: Mapping[str, Any]) -> dict[Any, str]:
    categories: dict[Any, str] = {}
    for row in data.get("categories", []):
        if isinstance(row, Mapping) and "id" in row:
            categories[row["id"]] = str(row.get("name", f"class_{row['id']}"))
    return categories


def _coco_images(data: Mapping[str, Any]) -> dict[Any, Mapping[str, Any]]:
    images: dict[Any, Mapping[str, Any]] = {}
    for row in data.get("images", []):
        if isinstance(row, Mapping) and "id" in row:
            images[row["id"]] = row
    return images


def _image_name_from_coco(image: Mapping[str, Any]) -> str:
    value = str(image.get("file_name") or image.get("image_name") or image.get("filename") or "")
    return value.replace("\\", "/")


def _read_model_class_csv(path: Path) -> dict[str, str]:
    classes: dict[str, str] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                classes[row[0].strip()] = row[1].strip()
    return classes


def _bbox_from_minmax(row: Mapping[str, str]) -> tuple[float, float, float, float] | None:
    try:
        xmin = float(row["xmin"])
        xmax = float(row["xmax"])
        ymin = float(row["ymin"])
        ymax = float(row["ymax"])
    except (KeyError, TypeError, ValueError):
        return None
    return xmin, ymin, xmax - xmin, ymax - ymin


def _coerce_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        return None
    try:
        x, y, width, height = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return x, y, width, height


def _bbox_from_xyxy(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        return None
    try:
        xmin, ymin, xmax, ymax = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    width = xmax - xmin
    height = ymax - ymin
    if width <= 0 or height <= 0:
        return None
    return xmin, ymin, width, height


def _bbox_from_image_json_object(obj: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    xyxy = _bbox_from_xyxy(obj.get("bbox"))
    xywh = _coerce_bbox(obj.get("bbox"))
    area = _float_or_none(obj.get("area"))
    if area is not None and xyxy is not None and xywh is not None:
        xyxy_error = abs((xyxy[2] * xyxy[3]) - area)
        xywh_error = abs((xywh[2] * xywh[3]) - area)
        return xyxy if xyxy_error <= xywh_error else xywh
    return xyxy or xywh


def _image_json_size(value: Any) -> tuple[int, int]:
    if not isinstance(value, Mapping):
        return 0, 0
    try:
        return int(float(value.get("width") or 0)), int(float(value.get("height") or 0))
    except (TypeError, ValueError):
        return 0, 0


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_yolo_rows(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows: list[tuple[int, float, float, float, float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(float(parts[0]))
            values = tuple(float(value) for value in parts[1:])
        except ValueError:
            continue
        rows.append((class_id, *values))
    return rows


def _image_name_for_yolo_label(label_path: Path, dataset_root: Path) -> str:
    split = label_path.parent.parent.name
    stem = label_path.stem
    image_dir = dataset_root / split / "images"
    for suffix in IMAGE_SUFFIXES:
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.exists():
            return _relative_or_name(candidate, dataset_root)
    return f"{split}/images/{stem}.jpg"


def _label_from_yolo_filename(path: Path) -> str | None:
    stem = path.stem
    if "--" in stem:
        return stem.split("--", 1)[0]
    if "_" in stem:
        parts = stem.split("_")
        if len(parts) >= 3 and parts[1] not in {"BOV", "F"}:
            return "_".join(parts[:-1])
    return None


def _yolo_bbox_to_xywh(
    x_center: float,
    y_center: float,
    box_width: float,
    box_height: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    if image_width <= 0 or image_height <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    width = box_width * image_width
    height = box_height * image_height
    x = (x_center - box_width / 2.0) * image_width
    y = (y_center - box_height / 2.0) * image_height
    return x, y, width, height


def _valid_review_box(annotation: ExternalAnnotation) -> bool:
    _, _, width, height = annotation.bbox
    return width > 0 and height > 0 and annotation.width > 0 and annotation.height > 0


def _valid_bbox(bbox: tuple[float, float, float, float] | None) -> bool:
    if bbox is None:
        return False
    _, _, width, height = bbox
    return width > 0 and height > 0


def _annotation_subfolder(annotation: ExternalAnnotation, dataset_root: Path) -> str:
    if annotation.source_format == "coco":
        return _coco_annotation_subfolder(annotation, dataset_root)

    image_path = _find_image_path(annotation.image_name, dataset_root)
    if image_path:
        return _relative_or_dot(image_path.parent, dataset_root)

    image_parent = Path(annotation.image_name.replace("\\", "/")).parent
    if image_parent != Path("."):
        return image_parent.as_posix()

    try:
        return _relative_or_dot(annotation.source_path.parent, dataset_root)
    except ValueError:
        return "."


def _coco_annotation_subfolder(annotation: ExternalAnnotation, dataset_root: Path) -> str:
    image_path = _find_coco_image_near_annotation(annotation.image_name, annotation.source_path, dataset_root)
    if image_path:
        subfolder = _sample_subfolder_for_image_path(image_path, annotation.source_path, dataset_root)
        if subfolder != ".":
            return subfolder

    image_parent = _strip_image_container_leaf(Path(annotation.image_name.replace("\\", "/")).parent)
    if image_parent != Path("."):
        return image_parent.as_posix()

    try:
        return _relative_or_dot(annotation.source_path.parent, dataset_root)
    except ValueError:
        return "."


def _find_coco_image_near_annotation(image_name: str, annotation_path: Path, dataset_root: Path) -> Path | None:
    value = Path(image_name.replace("\\", "/"))
    candidates: list[Path] = []

    if value.is_absolute():
        candidates.append(dataset_root / value.name)
    else:
        candidates.append(annotation_path.parent / value)
        if value.parent == Path("."):
            candidates.append(annotation_path.parent / "images" / value.name)
        candidates.append(dataset_root / value)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
            return candidate
    return None


def _sample_subfolder_for_image_path(image_path: Path, annotation_path: Path, dataset_root: Path) -> str:
    try:
        relative_parent = image_path.parent.relative_to(dataset_root)
    except ValueError:
        try:
            return _relative_or_dot(annotation_path.parent, dataset_root)
        except ValueError:
            return "."

    subfolder = _strip_image_container_leaf(relative_parent)
    if subfolder == Path("."):
        try:
            return _relative_or_dot(annotation_path.parent, dataset_root)
        except ValueError:
            return "."
    return subfolder.as_posix()


def _strip_image_container_leaf(path: Path) -> Path:
    if path.parts and path.parts[-1].lower() in {"image", "images", "imgs", "jpegimages"}:
        return Path(*path.parts[:-1])
    return path


def _relative_or_dot(path: Path, root: Path) -> str:
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    return relative or "."


def _image_size_for_name(image_name: str, dataset_root: Path) -> tuple[int, int]:
    image_path = _find_image_path(image_name, dataset_root)
    if not image_path:
        return 0, 0
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            return image.size
    except Exception:
        return 0, 0


def _review_image_name(image_name: str, dataset_root: Path, images_dir: Path) -> tuple[str, bool]:
    image_path = _find_image_path(image_name, images_dir) or _find_image_path(image_name, dataset_root)
    if image_path:
        try:
            return image_path.relative_to(images_dir).as_posix(), True
        except ValueError:
            return image_path.name, True
    path = Path(image_name)
    return path.name if path.is_absolute() else path.as_posix(), False


def _find_image_path(image_name: str, root: Path) -> Path | None:
    root = root.expanduser().resolve()
    value = image_name.replace("\\", "/")
    path = Path(value)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(root / path.name)
    else:
        candidates.append(root / path)
        candidates.append(root / path.name)
    for candidate in candidates:
        if candidate.exists() and candidate.suffix.lower() in IMAGE_SUFFIXES:
            return candidate

    for match in _image_index(root).get(path.name, []):
        if match.is_file():
            return match
    return None


def _image_index(root: Path) -> dict[str, list[Path]]:
    cached = _IMAGE_INDEX_CACHE.get(root)
    if cached is not None:
        return cached

    index: dict[str, list[Path]] = {}
    for match in root.rglob("*"):
        if match.is_file() and match.suffix.lower() in IMAGE_SUFFIXES:
            index.setdefault(match.name, []).append(match)
    _IMAGE_INDEX_CACHE[root] = index
    return index


def _relative_or_name(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _dataset_name_for_annotation(path: Path, external_root: Path) -> str:
    relative = path.relative_to(external_root)
    if relative.parts[0] == "cysts_dataset_all" and len(relative.parts) > 1:
        return f"{relative.parts[0]}/{relative.parts[1]}"
    return relative.parts[0]


def _dataset_root_for_annotation(path: Path, external_root: Path) -> Path:
    relative = path.relative_to(external_root)
    if relative.parts[0] == "cysts_dataset_all" and len(relative.parts) > 1:
        return external_root / relative.parts[0] / relative.parts[1]
    return external_root / relative.parts[0]


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _collect_raw_external_labels(external_root: Path) -> dict[str, set[str]]:
    labels: dict[str, set[str]] = {}

    for path in sorted(external_root.rglob("*.json")):
        if _is_ignored_external_path(path):
            continue
        try:
            data = _read_json(path)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue

        if _is_coco(data):
            for row in data.get("categories", []):
                if isinstance(row, Mapping):
                    _add_raw_label(labels, row.get("name"), path, external_root)
        elif _is_image_json_annotation(data):
            for obj in data.get("objects", []):
                if isinstance(obj, Mapping):
                    _add_raw_label(
                        labels,
                        obj.get("label") or obj.get("name") or obj.get("category"),
                        path,
                        external_root,
                    )
        elif isinstance(data, Mapping):
            _collect_via_raw_labels(labels, data, path, external_root)

    for classes_path in sorted(external_root.rglob("model_class.csv")):
        for label in _read_model_class_csv(classes_path).values():
            _add_raw_label(labels, label, classes_path, external_root)

    for xml_path in _annotation_files(external_root, "*.xml"):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            continue
        for obj in root.findall("object"):
            _add_raw_label(labels, _xml_text(obj, "name"), xml_path, external_root)

    for label_path in _annotation_files(external_root, "labels/*.txt"):
        label = _label_from_yolo_filename(label_path)
        class_names = _yolo_class_names_for_label_path(label_path, _top_level_external_dataset_root(label_path, external_root))
        if label:
            _add_raw_label(labels, label, label_path, external_root)
        else:
            for class_id, *_ in _read_yolo_rows(label_path):
                _add_raw_label(labels, class_names.get(class_id, f"class_{class_id}"), label_path, external_root)

    return labels


def _collect_via_raw_labels(
    labels: dict[str, set[str]],
    data: Mapping[str, Any],
    path: Path,
    external_root: Path,
) -> None:
    for image_row in data.values():
        if not isinstance(image_row, Mapping):
            continue
        regions = image_row.get("regions")
        if not isinstance(regions, list):
            continue
        for region in regions:
            if not isinstance(region, Mapping):
                continue
            attributes = region.get("region_attributes")
            if not isinstance(attributes, Mapping):
                continue
            _add_raw_label(labels, _via_region_label(attributes), path, external_root)


def _via_region_label(attributes: Mapping[str, Any]) -> str | None:
    for key in ("Cyst", "label", "class", "name"):
        value = attributes.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for value in attributes.values():
        if isinstance(value, str) and value.strip().lower() not in {"sure", "unsure"}:
            return value
    return None


def _add_raw_label(
    labels: dict[str, set[str]],
    label: Any,
    source_path: Path,
    external_root: Path,
) -> None:
    if label is None:
        return
    raw_label = str(label).strip()
    if not raw_label:
        return
    try:
        source = source_path.relative_to(external_root).as_posix()
    except ValueError:
        source = source_path.as_posix()
    labels.setdefault(raw_label, set()).add(source)


def _yolo_class_names_for_label_path(label_path: Path, dataset_root: Path) -> dict[int, str]:
    dataset_root = dataset_root.resolve()
    cache_key = (label_path.parent.resolve(), dataset_root)
    cached = _YOLO_LABEL_CLASS_NAMES_CACHE.get(cache_key)
    if cached is not None:
        return cached

    for parent in label_path.parents:
        if not _is_relative_to(parent.resolve(), dataset_root):
            break
        data_yaml = parent / "data.yaml"
        if data_yaml.exists():
            names = _read_yolo_data_yaml_names(data_yaml)
            _YOLO_LABEL_CLASS_NAMES_CACHE[cache_key] = names
            return names

    data_yamls = _yolo_data_yaml_paths(dataset_root)
    names = _read_yolo_data_yaml_names(data_yamls[0]) if len(data_yamls) == 1 else {}
    _YOLO_LABEL_CLASS_NAMES_CACHE[cache_key] = names
    return names


def _yolo_data_yaml_paths(dataset_root: Path) -> list[Path]:
    cached = _YOLO_DATA_YAML_PATHS_CACHE.get(dataset_root)
    if cached is not None:
        return cached
    paths = sorted(path for path in dataset_root.rglob("data.yaml") if path.is_file())
    _YOLO_DATA_YAML_PATHS_CACHE[dataset_root] = paths
    return paths


def _read_yolo_data_yaml_names(path: Path) -> dict[int, str]:
    cached = _YOLO_CLASS_NAMES_CACHE.get(path)
    if cached is not None:
        return cached

    names: dict[int, str] = {}
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    in_names_block = False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("names:"):
            in_names_block = True
            names.update(_parse_inline_yolo_names(stripped.split(":", 1)[1].strip()))
            if names:
                break
            continue
        if in_names_block:
            if stripped.startswith("-"):
                names[len(names)] = _clean_yaml_scalar(stripped[1:].strip())
                continue
            if re.match(r"^[A-Za-z_][A-Za-z0-9_-]*\s*:", stripped):
                break

    _YOLO_CLASS_NAMES_CACHE[path] = names
    return names


def _parse_inline_yolo_names(value: str) -> dict[int, str]:
    if not value:
        return {}
    if value.startswith("[") and value.endswith("]"):
        items = [item.strip() for item in value[1:-1].split(",")]
        return {idx: _clean_yaml_scalar(item) for idx, item in enumerate(items) if item}
    if value.startswith("{") and value.endswith("}"):
        names: dict[int, str] = {}
        for item in value[1:-1].split(","):
            if ":" not in item:
                continue
            key, raw_name = item.split(":", 1)
            try:
                class_id = int(_clean_yaml_scalar(key.strip()))
            except ValueError:
                continue
            names[class_id] = _clean_yaml_scalar(raw_name.strip())
        return names
    return {}


def _clean_yaml_scalar(value: str) -> str:
    return value.strip().strip("'\"")


def _top_level_external_dataset_root(path: Path, external_root: Path) -> Path:
    try:
        first_part = path.relative_to(external_root).parts[0]
    except (IndexError, ValueError):
        return external_root
    return external_root / first_part


def normalize_external_label(label: str) -> str:
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", label.strip())
    text = text.lower()
    text = text.replace("_", " ").replace("-", " ")
    text = text.replace(".", " ")
    for parenthesized in re.findall(r"\(([^)]*)\)", text):
        candidate = re.sub(r"\s+", " ", parenthesized).strip()
        if candidate in _LABEL_FIXES:
            return _LABEL_FIXES[candidate]
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\beggs?\b", " ", text)
    text = re.sub(r"\bspp?\.\b", "spp", text)
    text = re.sub(r"\bsp\.\b", "sp", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text in _LABEL_FIXES:
        return _LABEL_FIXES[text]
    normalized = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return normalized or "unknown"


def label_from_image_filename(filename: str, default: str = "parasite") -> str:
    """Infer a review label from an image filename.

    Existing flattened project files use microscope__parasite__hash names. Other
    flat external folders, such as NAING, store the parasite name directly in the
    filename before a numeric parenthetical suffix.
    """
    stem = Path(str(filename).replace("\\", "/")).stem.strip()
    parts = stem.split("__")
    if len(parts) >= 2 and parts[1].strip():
        return parts[1].replace("_", " ").lower()

    label = re.sub(r"\s*\(\s*\d+\s*\)\s*$", "", stem).strip()
    if not label:
        return default
    normalized = normalize_external_label(label)
    return default if normalized == "unknown" else normalized
