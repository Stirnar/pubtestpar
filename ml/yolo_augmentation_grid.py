from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


DEFAULT_AUGMENTATION_STRATEGIES = (
    "translation",
    "scale",
    "rotation",
    "flip",
    "brightness_contrast",
    "hsv_color",
)

AGGRESSIVE_AUGMENTATION_STRATEGIES = (
    *DEFAULT_AUGMENTATION_STRATEGIES,
    "shear",
    "perspective",
    "mosaic",
    "mixup",
)

AUGMENTATION_GRID_PRESETS = ("separate", "aggressive", "compact")

COMPACT_AUGMENTATION_GRID = (
    ("hsv_color", 4),
    ("scale", 4),
    ("ood_color", 2),
    ("ood_color", 4),
    ("ood_geometry", 2),
    ("ood_geometry", 4),
    ("ood_acquisition", 2),
    ("ood_acquisition", 3),
    ("ood_acquisition", 4),
    ("mosaic_mix", 2),
    ("mosaic_mix", 4),
)

NO_AUGMENTATION_PARAMS: dict[str, float] = {
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
}

_MAX_INTENSITY = 4


@dataclass(frozen=True)
class YoloExperiment:
    name: str
    profile: str
    intensity: int | None
    description: str
    augmentation: dict[str, Any]


def build_yolo_experiment_plan(
    max_intensity: int,
    strategies: Sequence[str] | None = None,
    include_default: bool = True,
    include_no_augmentation: bool = True,
    run_prefix: str | None = None,
    grid_preset: str = "separate",
) -> list[YoloExperiment]:
    if max_intensity < 1 or max_intensity > _MAX_INTENSITY:
        raise ValueError("--max-intensity must be 1, 2, 3, or 4.")
    if grid_preset not in AUGMENTATION_GRID_PRESETS:
        raise ValueError(f"Unknown augmentation grid preset: {grid_preset}")

    if grid_preset == "compact":
        if strategies:
            raise ValueError("--strategy cannot be used with --grid-preset compact.")
        experiments = _compact_ood_experiments(include_no_augmentation, include_default, run_prefix)
        _check_unique_experiment_names(experiments)
        return experiments

    selected_strategies = strategies or default_strategies_for_preset(grid_preset)
    experiments = _baseline_experiments(include_no_augmentation, include_default, run_prefix)
    experiments.extend(_single_strategy_experiments(selected_strategies, max_intensity, run_prefix))
    if grid_preset == "aggressive":
        experiments.extend(_combined_ood_experiments(max_intensity, run_prefix))

    _check_unique_experiment_names(experiments)
    return experiments


def default_strategies_for_preset(grid_preset: str) -> tuple[str, ...]:
    if grid_preset == "aggressive":
        return AGGRESSIVE_AUGMENTATION_STRATEGIES
    if grid_preset == "separate":
        return DEFAULT_AUGMENTATION_STRATEGIES
    if grid_preset == "compact":
        return ()
    raise ValueError(f"Unknown augmentation grid preset: {grid_preset}")


def _baseline_experiments(
    include_no_augmentation: bool,
    include_default: bool,
    run_prefix: str | None,
) -> list[YoloExperiment]:
    experiments: list[YoloExperiment] = []
    if include_no_augmentation:
        experiments.append(
            _experiment(
                "no_aug",
                None,
                "All YOLO augmentation parameters disabled.",
                NO_AUGMENTATION_PARAMS,
                run_prefix,
            )
        )
    if include_default:
        experiments.append(
            _experiment(
                "default",
                None,
                "Ultralytics default training configuration.",
                {},
                run_prefix,
            )
        )
    return experiments


def _single_strategy_experiments(
    strategies: Sequence[str],
    max_intensity: int,
    run_prefix: str | None,
) -> list[YoloExperiment]:
    experiments: list[YoloExperiment] = []
    for strategy in strategies:
        for intensity in range(1, max_intensity + 1):
            params, description = _single_strategy_params(strategy, intensity)
            experiments.append(_experiment(strategy, intensity, description, params, run_prefix))
    return experiments


def _single_strategy_params(strategy: str, intensity: int) -> tuple[dict[str, Any], str]:
    params = dict(NO_AUGMENTATION_PARAMS)
    index = intensity - 1
    if strategy == "translation":
        params["translate"] = [0.05, 0.10, 0.20, 0.30][index]
        description = "Only translation augmentation enabled."
    elif strategy == "scale":
        params["scale"] = [0.10, 0.25, 0.50, 0.75][index]
        description = "Only scale augmentation enabled."
    elif strategy == "rotation":
        params["degrees"] = [5.0, 10.0, 20.0, 35.0][index]
        description = "Only rotation augmentation enabled."
    elif strategy == "flip":
        probability = [0.25, 0.50, 0.75, 0.90][index]
        params["fliplr"] = probability
        params["flipud"] = probability
        description = "Only horizontal and vertical flip augmentation enabled."
    elif strategy == "brightness_contrast":
        params["hsv_v"] = [0.10, 0.25, 0.40, 0.70][index]
        description = "Only YOLO brightness/value augmentation enabled."
    elif strategy == "hsv_color":
        params["hsv_h"] = [0.005, 0.010, 0.015, 0.030][index]
        params["hsv_s"] = [0.20, 0.45, 0.70, 0.95][index]
        params["hsv_v"] = [0.10, 0.25, 0.40, 0.70][index]
        description = "Only HSV color augmentation enabled."
    elif strategy == "shear":
        params["shear"] = [2.0, 5.0, 10.0, 15.0][index]
        description = "Only shear augmentation enabled."
    elif strategy == "perspective":
        params["perspective"] = [0.00025, 0.0005, 0.00075, 0.001][index]
        description = "Only perspective augmentation enabled."
    elif strategy == "mosaic":
        params["mosaic"] = [0.25, 0.50, 0.75, 1.0][index]
        description = "Only mosaic augmentation enabled."
    elif strategy == "mixup":
        params["mixup"] = [0.05, 0.10, 0.20, 0.30][index]
        description = "Only mixup augmentation enabled."
    else:
        raise ValueError(f"Unknown augmentation strategy: {strategy}")
    return params, description


def _combined_ood_experiments(max_intensity: int, run_prefix: str | None) -> list[YoloExperiment]:
    experiments: list[YoloExperiment] = []
    for intensity in range(1, max_intensity + 1):
        for profile, description, params in _combined_ood_profiles(intensity):
            experiments.append(_experiment(profile, intensity, description, params, run_prefix))
    return experiments


def _compact_ood_experiments(
    include_no_augmentation: bool,
    include_default: bool,
    run_prefix: str | None,
) -> list[YoloExperiment]:
    experiments = _baseline_experiments(include_no_augmentation, include_default, run_prefix)
    for profile, intensity in COMPACT_AUGMENTATION_GRID:
        params, description = _profile_params(profile, intensity)
        experiments.append(_experiment(profile, intensity, description, params, run_prefix))
    return experiments


def _profile_params(profile: str, intensity: int) -> tuple[dict[str, Any], str]:
    if profile in AGGRESSIVE_AUGMENTATION_STRATEGIES:
        return _single_strategy_params(profile, intensity)
    for candidate_profile, description, params in _combined_ood_profiles(intensity):
        if candidate_profile == profile:
            return params, description
    raise ValueError(f"Unknown augmentation profile: {profile}")


def _combined_ood_profiles(intensity: int) -> list[tuple[str, str, dict[str, Any]]]:
    color = _ood_color_params(intensity)
    geometry = _ood_geometry_params(intensity)
    mosaic_mix = _mosaic_mix_params(intensity)
    acquisition = _merge_params(color, geometry, mosaic_mix)
    return [
        ("ood_color", "Combined HSV and illumination stress for acquisition differences.", color),
        ("ood_geometry", "Combined geometry stress: rotate, shift, scale, shear, perspective, and flips.", geometry),
        ("mosaic_mix", "Combined mosaic and mixup stress with other augmentation disabled.", mosaic_mix),
        ("ood_acquisition", "Combined aggressive color, illumination, geometry, mosaic, and mixup stress.", acquisition),
    ]


def _ood_color_params(intensity: int) -> dict[str, Any]:
    params = dict(NO_AUGMENTATION_PARAMS)
    index = intensity - 1
    params["hsv_h"] = [0.010, 0.015, 0.025, 0.040][index]
    params["hsv_s"] = [0.35, 0.60, 0.85, 1.00][index]
    params["hsv_v"] = [0.25, 0.45, 0.65, 0.85][index]
    return params


def _ood_geometry_params(intensity: int) -> dict[str, Any]:
    params = dict(NO_AUGMENTATION_PARAMS)
    index = intensity - 1
    params["degrees"] = [10.0, 20.0, 35.0, 50.0][index]
    params["translate"] = [0.10, 0.15, 0.25, 0.35][index]
    params["scale"] = [0.25, 0.45, 0.70, 0.90][index]
    params["shear"] = [2.0, 5.0, 10.0, 15.0][index]
    params["perspective"] = [0.00025, 0.0005, 0.00075, 0.001][index]
    params["fliplr"] = [0.25, 0.50, 0.75, 0.90][index]
    params["flipud"] = [0.25, 0.50, 0.75, 0.90][index]
    return params


def _mosaic_mix_params(intensity: int) -> dict[str, Any]:
    params = dict(NO_AUGMENTATION_PARAMS)
    index = intensity - 1
    params["mosaic"] = [0.25, 0.50, 0.75, 1.0][index]
    params["mixup"] = [0.05, 0.10, 0.20, 0.30][index]
    return params


def _merge_params(*param_sets: dict[str, Any]) -> dict[str, Any]:
    merged = dict(NO_AUGMENTATION_PARAMS)
    for params in param_sets:
        for key, value in params.items():
            if key not in NO_AUGMENTATION_PARAMS or value != NO_AUGMENTATION_PARAMS[key]:
                merged[key] = value
    return merged


def _experiment(
    profile: str,
    intensity: int | None,
    description: str,
    params: dict[str, Any],
    run_prefix: str | None,
) -> YoloExperiment:
    suffix = profile if intensity is None else f"{profile}_i{intensity}"
    name = f"{run_prefix}_{suffix}" if run_prefix else suffix
    return YoloExperiment(
        name=name,
        profile=profile,
        intensity=intensity,
        description=description,
        augmentation=dict(params),
    )


def _check_unique_experiment_names(experiments: Sequence[YoloExperiment]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for experiment in experiments:
        if experiment.name in seen:
            duplicates.append(experiment.name)
        seen.add(experiment.name)
    if duplicates:
        names = ", ".join(sorted(set(duplicates)))
        raise ValueError(f"Duplicate experiment names in augmentation grid: {names}")
