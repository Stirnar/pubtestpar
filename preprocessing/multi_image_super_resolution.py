from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MOTION_MODELS = ("translation", "euclidean", "affine")
METRIC_FIELDS = [
    "laplacian_variance",
    "tenengrad",
    "brenner",
    "high_frequency_ratio",
    "edge_density",
]


@dataclass
class ImageRecord:
    path: Path
    image: "np.ndarray"
    gray: "np.ndarray"
    metrics: Dict[str, float]
    fusion_weight: float = 1.0
    registration_cc: Optional[float] = None
    registration_ok: bool = False
    warp_matrix: Optional["np.ndarray"] = None


def require_numpy() -> None:
    if np is None:
        raise ModuleNotFoundError("NumPy is required for this script. Install it with: pip install numpy")


def require_cv2() -> None:
    if cv2 is None:
        raise ModuleNotFoundError(
            "OpenCV is required for this script. Install it with: pip install opencv-python-headless"
        )


def require_runtime_dependencies() -> None:
    require_numpy()
    require_cv2()


def motion_model_code(name: str) -> int:
    require_cv2()
    motion_codes = {
        "translation": cv2.MOTION_TRANSLATION,
        "euclidean": cv2.MOTION_EUCLIDEAN,
        "affine": cv2.MOTION_AFFINE,
    }
    return motion_codes[name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone multi-image super-resolution for microscope images. "
            "This script aligns related images from one folder and fuses them into a higher-resolution output."
        )
    )
    parser.add_argument("input_dir", type=str, help="Directory containing related images of the same scene.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for the fused image and metrics. Defaults to <input_dir>/super_resolution_output.",
    )
    parser.add_argument("--scale", type=int, default=2, help="Upsampling factor for the fused output.")
    parser.add_argument(
        "--motion-model",
        choices=sorted(MOTION_MODELS),
        default="euclidean",
        help="Registration model used during alignment.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optionally keep only the sharpest K input images before fusion.",
    )
    parser.add_argument(
        "--ecc-iterations",
        type=int,
        default=300,
        help="Maximum iterations for ECC registration.",
    )
    parser.add_argument(
        "--ecc-eps",
        type=float,
        default=1e-5,
        help="ECC convergence epsilon.",
    )
    parser.add_argument(
        "--sharpen-amount",
        type=float,
        default=0.5,
        help="Strength of the final unsharp-mask enhancement.",
    )
    parser.add_argument(
        "--sharpen-sigma",
        type=float,
        default=1.2,
        help="Gaussian sigma used by the final unsharp mask.",
    )
    parser.add_argument(
        "--save-aligned",
        action="store_true",
        help="Also save the low-resolution aligned inputs used during fusion.",
    )
    return parser.parse_args()


def list_image_paths(input_dir: Path) -> List[Path]:
    return sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def image_to_float32(image: np.ndarray) -> np.ndarray:
    require_numpy()
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    if image.dtype == np.uint16:
        return image.astype(np.float32) / 65535.0

    image = image.astype(np.float32)
    max_value = float(np.max(image)) if image.size else 1.0
    if max_value > 1.0:
        image = image / max_value
    return np.clip(image, 0.0, 1.0)


def load_image(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    require_runtime_dependencies()
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {path}")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    image = image_to_float32(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray


def resize_if_needed(image: np.ndarray, target_size: Tuple[int, int], interpolation: int) -> np.ndarray:
    require_runtime_dependencies()
    target_w, target_h = target_size
    if image.shape[1] == target_w and image.shape[0] == target_h:
        return image
    return cv2.resize(image, target_size, interpolation=interpolation)


def compute_high_frequency_ratio(gray: np.ndarray) -> float:
    require_numpy()
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    power = np.abs(spectrum) ** 2

    h, w = gray.shape
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    cutoff = 0.15 * min(h, w)

    high_frequency_energy = float(power[radius >= cutoff].sum())
    total_energy = float(power.sum()) + 1e-8
    return high_frequency_energy / total_energy


def compute_resolution_metrics(image: np.ndarray) -> Dict[str, float]:
    require_runtime_dependencies()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = gray.astype(np.float32)

    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude_sq = grad_x * grad_x + grad_y * grad_y

    if gray.shape[1] > 2:
        brenner = np.mean((gray[:, 2:] - gray[:, :-2]) ** 2)
    else:
        brenner = 0.0

    edge_map = cv2.Canny(np.clip(gray * 255.0, 0, 255).astype(np.uint8), 100, 200)

    return {
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "laplacian_variance": float(laplacian.var()),
        "tenengrad": float(np.mean(gradient_magnitude_sq)),
        "brenner": float(brenner),
        "high_frequency_ratio": float(compute_high_frequency_ratio(gray)),
        "edge_density": float(np.mean(edge_map > 0)),
    }


def load_records(paths: Iterable[Path]) -> List[ImageRecord]:
    require_runtime_dependencies()
    records: List[ImageRecord] = []
    for path in paths:
        image, gray = load_image(path)
        metrics = compute_resolution_metrics(image)
        records.append(ImageRecord(path=path, image=image, gray=gray, metrics=metrics))
    return records


def initial_quality_score(metrics: Dict[str, float]) -> float:
    require_numpy()
    return (
        0.45 * np.log1p(metrics["laplacian_variance"])
        + 0.30 * np.log1p(metrics["tenengrad"])
        + 0.15 * np.log1p(metrics["brenner"])
        + 0.10 * metrics["high_frequency_ratio"]
    )


def select_records(records: List[ImageRecord], top_k: Optional[int]) -> List[ImageRecord]:
    ranked = sorted(records, key=lambda record: initial_quality_score(record.metrics), reverse=True)
    if top_k is None or top_k >= len(ranked):
        return ranked
    return ranked[:top_k]


def normalize_for_registration(gray: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    require_runtime_dependencies()
    resized = resize_if_needed(gray, target_size, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    return cv2.normalize(blurred, None, 0.0, 1.0, cv2.NORM_MINMAX).astype(np.float32)


def register_to_reference(
    moving_gray: np.ndarray,
    reference_gray: np.ndarray,
    motion_model: str,
    iterations: int,
    eps: float,
) -> Tuple[np.ndarray, Optional[float], bool]:
    require_runtime_dependencies()
    motion = motion_model_code(motion_model)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, eps)

    try:
        cc, warp = cv2.findTransformECC(reference_gray, moving_gray, warp, motion, criteria)
        return warp.astype(np.float32), float(cc), True
    except cv2.error:
        return np.eye(2, 3, dtype=np.float32), None, False


def scale_warp_matrix(warp: np.ndarray, scale: int) -> np.ndarray:
    require_numpy()
    scaled = warp.copy().astype(np.float32)
    scaled[:, 2] *= float(scale)
    return scaled


def warp_image(
    image: np.ndarray,
    warp: np.ndarray,
    size: Tuple[int, int],
    interpolation: int,
    border_mode: int,
    border_value: float = 0.0,
) -> np.ndarray:
    require_runtime_dependencies()
    return cv2.warpAffine(
        image,
        warp,
        size,
        flags=interpolation | cv2.WARP_INVERSE_MAP,
        borderMode=border_mode,
        borderValue=border_value,
    )


def unsharp_mask(image: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    require_runtime_dependencies()
    if amount <= 0:
        return np.clip(image, 0.0, 1.0)

    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharpened = image * (1.0 + amount) - blurred * amount
    return np.clip(sharpened, 0.0, 1.0)


def fuse_records(
    records: List[ImageRecord],
    reference_record: ImageRecord,
    scale: int,
    motion_model: str,
    iterations: int,
    eps: float,
    sharpen_amount: float,
    sharpen_sigma: float,
    output_dir: Path,
    save_aligned: bool,
) -> np.ndarray:
    require_runtime_dependencies()
    ref_h, ref_w = reference_record.gray.shape
    target_lr_size = (ref_w, ref_h)
    target_hr_size = (ref_w * scale, ref_h * scale)

    reference_registration = normalize_for_registration(reference_record.gray, target_lr_size)
    aligned_dir = output_dir / "aligned_inputs"
    if save_aligned:
        aligned_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        resized_gray = normalize_for_registration(record.gray, target_lr_size)
        if record.path == reference_record.path:
            record.warp_matrix = np.eye(2, 3, dtype=np.float32)
            record.registration_cc = 1.0
            record.registration_ok = True
        else:
            warp, cc, ok = register_to_reference(
                resized_gray,
                reference_registration,
                motion_model=motion_model,
                iterations=iterations,
                eps=eps,
            )
            record.warp_matrix = warp
            record.registration_cc = cc
            record.registration_ok = ok

        record.fusion_weight = max(initial_quality_score(record.metrics), 1e-6)
        if not record.registration_ok:
            record.fusion_weight *= 0.25

        if save_aligned:
            aligned_lr = warp_image(
                resize_if_needed(record.image, target_lr_size, interpolation=cv2.INTER_CUBIC),
                record.warp_matrix,
                target_lr_size,
                interpolation=cv2.INTER_CUBIC,
                border_mode=cv2.BORDER_REFLECT,
            )
            write_image(aligned_dir / f"{record.path.stem}_aligned.png", aligned_lr)

    total_weight = sum(record.fusion_weight for record in records) or 1.0
    for record in records:
        record.fusion_weight /= total_weight

    accumulator = np.zeros((target_hr_size[1], target_hr_size[0], 3), dtype=np.float32)
    weight_sum = np.zeros((target_hr_size[1], target_hr_size[0], 1), dtype=np.float32)

    unit_mask_lr = np.ones((ref_h, ref_w), dtype=np.float32)
    unit_mask_hr = cv2.resize(unit_mask_lr, target_hr_size, interpolation=cv2.INTER_NEAREST)

    for record in records:
        resized_color = resize_if_needed(record.image, target_lr_size, interpolation=cv2.INTER_CUBIC)
        upscaled = cv2.resize(resized_color, target_hr_size, interpolation=cv2.INTER_CUBIC)
        scaled_warp = scale_warp_matrix(record.warp_matrix, scale)
        aligned_hr = warp_image(
            upscaled,
            scaled_warp,
            target_hr_size,
            interpolation=cv2.INTER_CUBIC,
            border_mode=cv2.BORDER_REFLECT,
        )

        aligned_mask = warp_image(
            unit_mask_hr,
            scaled_warp,
            target_hr_size,
            interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT,
            border_value=0.0,
        )
        aligned_mask = np.clip(aligned_mask[..., None], 0.0, 1.0)
        weighted_mask = aligned_mask * record.fusion_weight

        accumulator += aligned_hr * weighted_mask
        weight_sum += weighted_mask

    fused = accumulator / np.clip(weight_sum, 1e-6, None)
    fused = unsharp_mask(fused, sigma=sharpen_sigma, amount=sharpen_amount)
    return np.clip(fused, 0.0, 1.0)


def write_image(path: Path, image: np.ndarray) -> None:
    require_runtime_dependencies()
    path.parent.mkdir(parents=True, exist_ok=True)
    image_8bit = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), image_8bit):
        raise ValueError(f"Could not write image to {path}")


def _normalize_series(values: List[float], use_log: bool) -> List[float]:
    require_numpy()
    array = np.asarray(values, dtype=np.float64)
    if use_log:
        array = np.log1p(np.clip(array, 0.0, None))

    min_value = float(array.min())
    max_value = float(array.max())
    if np.isclose(min_value, max_value):
        return [1.0 for _ in values]
    normalized = (array - min_value) / (max_value - min_value)
    return [float(v) for v in normalized]


def attach_resolution_scores(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    metric_weights = {
        "laplacian_variance": 0.30,
        "tenengrad": 0.30,
        "brenner": 0.20,
        "high_frequency_ratio": 0.10,
        "edge_density": 0.10,
    }
    log_scaled_metrics = {"laplacian_variance", "tenengrad", "brenner"}

    normalized_metrics = {}
    for metric_name in metric_weights:
        normalized_metrics[metric_name] = _normalize_series(
            [row[metric_name] for row in rows],
            use_log=metric_name in log_scaled_metrics,
        )

    for index, row in enumerate(rows):
        score = 0.0
        for metric_name, weight in metric_weights.items():
            score += weight * normalized_metrics[metric_name][index]
        row["resolution_score"] = float(score)

    return sorted(rows, key=lambda row: row["resolution_score"], reverse=True)


def percent_delta(new_value: float, baseline: float) -> Optional[float]:
    require_numpy()
    if np.isclose(baseline, 0.0):
        return None
    return float(100.0 * (new_value - baseline) / baseline)


def build_metric_rows(records: List[ImageRecord], output_image: np.ndarray) -> List[Dict[str, float]]:
    rows = []
    for record in records:
        row = {
            "name": record.path.name,
            "role": "input",
            "registration_ok": record.registration_ok,
            "registration_cc": record.registration_cc if record.registration_cc is not None else "",
            "fusion_weight": record.fusion_weight,
        }
        row.update(record.metrics)
        rows.append(row)

    output_metrics = compute_resolution_metrics(output_image)
    output_row = {
        "name": "super_resolved_output",
        "role": "output",
        "registration_ok": True,
        "registration_cc": 1.0,
        "fusion_weight": 1.0,
    }
    output_row.update(output_metrics)
    rows.append(output_row)
    return attach_resolution_scores(rows)


def write_metrics_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    fieldnames = [
        "name",
        "role",
        "width",
        "height",
        "laplacian_variance",
        "tenengrad",
        "brenner",
        "high_frequency_ratio",
        "edge_density",
        "resolution_score",
        "registration_ok",
        "registration_cc",
        "fusion_weight",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_summary(
    selected_records: List[ImageRecord],
    reference_record: ImageRecord,
    rows: List[Dict[str, float]],
    scale: int,
    output_image_path: Path,
) -> Dict[str, object]:
    require_numpy()
    input_rows = [row for row in rows if row["role"] == "input"]
    output_row = next(row for row in rows if row["role"] == "output")
    best_input = max(input_rows, key=lambda row: row["resolution_score"])

    mean_input_metrics = {
        metric: float(np.mean([row[metric] for row in input_rows])) for metric in METRIC_FIELDS + ["resolution_score"]
    }

    improvement_vs_mean = {
        metric: percent_delta(output_row[metric], mean_input_metrics[metric]) for metric in METRIC_FIELDS + ["resolution_score"]
    }
    improvement_vs_best = {
        metric: percent_delta(output_row[metric], best_input[metric]) for metric in METRIC_FIELDS + ["resolution_score"]
    }

    return {
        "input_count": len(selected_records),
        "reference_image": reference_record.path.name,
        "scale_factor": scale,
        "output_image": str(output_image_path),
        "notes": (
            "Resolution metrics are no-reference proxies based on sharpness, edges, and high-frequency energy. "
            "They are intended for relative comparison, not as a substitute for optical ground truth."
        ),
        "registrations": [
            {
                "image": record.path.name,
                "registration_ok": record.registration_ok,
                "registration_cc": record.registration_cc,
                "fusion_weight": record.fusion_weight,
            }
            for record in selected_records
        ],
        "comparison": {
            "best_input_image": best_input["name"],
            "output_metrics": {metric: output_row[metric] for metric in METRIC_FIELDS + ["resolution_score"]},
            "mean_input_metrics": mean_input_metrics,
            "best_input_metrics": {metric: best_input[metric] for metric in METRIC_FIELDS + ["resolution_score"]},
            "improvement_vs_mean_percent": improvement_vs_mean,
            "improvement_vs_best_percent": improvement_vs_best,
        },
        "ranking": [
            {
                "rank": index + 1,
                "name": row["name"],
                "role": row["role"],
                "resolution_score": row["resolution_score"],
            }
            for index, row in enumerate(rows)
        ],
    }


def save_summary(path: Path, summary: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    args = parse_args()
    require_runtime_dependencies()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    image_paths = list_image_paths(input_dir)
    if len(image_paths) < 2:
        raise ValueError("At least two images are required for multi-image super-resolution.")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (input_dir / "super_resolution_output").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records = load_records(image_paths)
    selected_records = select_records(all_records, args.top_k)
    reference_record = max(selected_records, key=lambda record: initial_quality_score(record.metrics))

    fused = fuse_records(
        records=selected_records,
        reference_record=reference_record,
        scale=args.scale,
        motion_model=args.motion_model,
        iterations=args.ecc_iterations,
        eps=args.ecc_eps,
        sharpen_amount=args.sharpen_amount,
        sharpen_sigma=args.sharpen_sigma,
        output_dir=output_dir,
        save_aligned=args.save_aligned,
    )

    output_image_path = output_dir / f"super_resolved_x{args.scale}.png"
    write_image(output_image_path, fused)

    metric_rows = build_metric_rows(selected_records, fused)
    metrics_csv_path = output_dir / "resolution_metrics.csv"
    write_metrics_csv(metrics_csv_path, metric_rows)

    summary = build_summary(
        selected_records=selected_records,
        reference_record=reference_record,
        rows=metric_rows,
        scale=args.scale,
        output_image_path=output_image_path,
    )
    summary_path = output_dir / "super_resolution_summary.json"
    save_summary(summary_path, summary)

    print(f"Saved fused image: {output_image_path}")
    print(f"Saved metric comparison: {metrics_csv_path}")
    print(f"Saved summary report: {summary_path}")


if __name__ == "__main__":
    main()
