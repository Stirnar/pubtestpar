from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ml.yolo_training import eval_yolo_from_namespace


@dataclass(frozen=True)
class ExperimentEvalJob:
    experiment_dir: Path
    model_path: Path
    eval_dir: Path
    output_parent: Path
    run_name: str
    data: str


@dataclass(frozen=True)
class ExperimentEvalBatchResult:
    experiments_dir: Path
    eval_root: Path
    total_models: int
    evaluated: int
    skipped_existing: int
    dry_run: bool
    skipped_eval_dirs: tuple[Path, ...]


def eval_missing_experiments_from_namespace(args: Any) -> ExperimentEvalBatchResult:
    experiments_dir = Path(args.experiments_dir).expanduser().resolve()
    eval_root = Path(args.eval_dir).expanduser().resolve()
    jobs, skipped_eval_dirs = plan_missing_experiment_evals(
        experiments_dir=experiments_dir,
        eval_root=eval_root,
        data=args.data,
    )

    for job in jobs:
        prefix = "Would evaluate" if args.dry_run else "Evaluating"
        print(f"{prefix}: {job.experiment_dir} -> {job.eval_dir}")
        if args.dry_run:
            continue
        job.output_parent.mkdir(parents=True, exist_ok=True)
        eval_yolo_from_namespace(
            SimpleNamespace(
                model=str(job.model_path),
                data=job.data,
                split=args.split,
                output_dir=str(job.output_parent),
                run_name=job.run_name,
                upload_dir=None,
                work_dir=args.work_dir,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                overwrite=False,
            )
        )

    return ExperimentEvalBatchResult(
        experiments_dir=experiments_dir,
        eval_root=eval_root,
        total_models=len(jobs) + len(skipped_eval_dirs),
        evaluated=len(jobs),
        skipped_existing=len(skipped_eval_dirs),
        dry_run=bool(args.dry_run),
        skipped_eval_dirs=tuple(skipped_eval_dirs),
    )


def summarize_experiment_eval_batch(result: ExperimentEvalBatchResult) -> str:
    action = "Would evaluate" if result.dry_run else "Evaluated"
    lines = [
        f"Experiments: {result.experiments_dir}",
        f"Eval root: {result.eval_root}",
        f"Best models found: {result.total_models}",
        f"{action}: {result.evaluated}",
        f"Skipped existing eval folders: {result.skipped_existing}",
    ]
    if result.total_models == 0:
        lines.append(f"No weights/best.pt files were found under {result.experiments_dir}.")
    elif result.evaluated == 0 and result.skipped_existing:
        lines.append("No missing evaluations remain; every discovered best.pt already has a matching eval folder.")
    if result.skipped_eval_dirs:
        sample = ", ".join(path.as_posix() for path in result.skipped_eval_dirs[:5])
        if len(result.skipped_eval_dirs) > 5:
            sample += ", ..."
        lines.append(f"Skipped folders: {sample}")
    return "\n".join(lines)


def plan_missing_experiment_evals(
    experiments_dir: Path,
    eval_root: Path,
    data: str | None = None,
) -> tuple[list[ExperimentEvalJob], list[Path]]:
    if not experiments_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiments_dir}")

    jobs: list[ExperimentEvalJob] = []
    skipped_eval_dirs: list[Path] = []
    for run_dir, model_path in _experiment_model_paths(experiments_dir):
        relative_run_dir = _relative_run_dir(run_dir, experiments_dir)
        eval_dir = eval_root / relative_run_dir
        if eval_dir.exists():
            skipped_eval_dirs.append(eval_dir)
            continue
        jobs.append(
            ExperimentEvalJob(
                experiment_dir=run_dir,
                model_path=model_path,
                eval_dir=eval_dir,
                output_parent=eval_dir.parent,
                run_name=eval_dir.name,
                data=data or _infer_dataset_yaml(run_dir, experiments_dir),
            )
        )
    return jobs, skipped_eval_dirs


def _experiment_model_paths(experiments_dir: Path) -> list[tuple[Path, Path]]:
    models: list[tuple[Path, Path]] = []
    for model_path in sorted(experiments_dir.rglob("weights/best.pt")):
        run_dir = model_path.parent.parent
        if run_dir.is_dir():
            models.append((run_dir, model_path))
    return models


def _relative_run_dir(run_dir: Path, experiments_dir: Path) -> Path:
    if run_dir == experiments_dir:
        return Path(run_dir.name)
    return run_dir.relative_to(experiments_dir)


def _infer_dataset_yaml(run_dir: Path, experiments_dir: Path) -> str:
    experiment_config = run_dir / "experiment_config.json"
    data = _data_from_json(experiment_config, ("yolo_args", "data"))
    if data:
        return data

    for parent in [run_dir, *run_dir.parents]:
        if parent == experiments_dir.parent:
            break
        data = _data_from_json(parent / "experiment_manifest.json", ("data",))
        if data:
            return data
        if parent == experiments_dir:
            break

    raise ValueError(
        f"Could not infer dataset.yaml for {run_dir}. "
        "Pass --data explicitly or keep experiment_manifest.json/experiment_config.json with the run."
    )


def _data_from_json(path: Path, keys: tuple[str, ...]) -> str | None:
    if not path.exists():
        return None
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return str(value) if value else None
