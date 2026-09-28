import argparse
import os
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import List, Optional

TMP_CACHE_ROOT = Path("/tmp/parasites_cache")
TMP_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
MPL_CACHE_DIR = TMP_CACHE_ROOT / "matplotlib"
XDG_CACHE_DIR = TMP_CACHE_ROOT / "xdg"
SKIMAGE_CACHE_DIR = TMP_CACHE_ROOT / "skimage"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
SKIMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))
os.environ.setdefault("SKIMAGE_DATADIR", str(SKIMAGE_CACHE_DIR))

import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_ubyte, io
from skimage.measure import shannon_entropy


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass
class RankedTile:
    entropy: float
    image: np.ndarray
    tile_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank existing tile images by entropy and display the best ones."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to a tile image or a directory containing tile images.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="How many highest-entropy tiles to show. Use 0 to show all ranked tiles. Default: 12.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=4,
        help="Number of columns in the ranking figure. Default: 4.",
    )
    parser.add_argument(
        "--save-figure",
        type=str,
        default=None,
        help="Optional output path for the ranking figure.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive matplotlib window.",
    )
    return parser.parse_args()


def iter_image_paths(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    image_paths = sorted(
        path for path in input_path.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported image files were found in: {input_path}")
    return image_paths


def load_image(image_path: Path) -> np.ndarray:
    image = io.imread(image_path, as_gray=True)
    image = np.asarray(image, dtype=np.float32)
    image = np.nan_to_num(image, copy=False)

    if image.size == 0:
        raise ValueError("image is empty")

    image_min = float(np.min(image))
    image_max = float(np.max(image))
    if image_max > image_min:
        image = (image - image_min) / (image_max - image_min)
    else:
        image = np.zeros_like(image, dtype=np.float32)

    return image


def compute_entropy(image: np.ndarray) -> float:
    tile_uint8 = img_as_ubyte(image)
    return float(shannon_entropy(tile_uint8))


def rank_tiles(image_paths: List[Path]) -> List[RankedTile]:
    ranked_tiles: List[RankedTile] = []
    for image_path in image_paths:
        try:
            image = load_image(image_path)
        except Exception as exc:
            print(f"Skipping {image_path.name}: {exc}")
            continue
        ranked_tiles.append(
            RankedTile(
                entropy=compute_entropy(image),
                image=image,
                tile_path=image_path,
            )
        )

    ranked_tiles.sort(key=lambda item: item.entropy, reverse=True)
    return ranked_tiles


def print_ranking(ranked_tiles: List[RankedTile]) -> None:
    print(f"\nRanked {len(ranked_tiles)} tiles from highest entropy to lowest:\n")
    for rank, item in enumerate(ranked_tiles, start=1):
        print(f"{rank:>3}. entropy={item.entropy:0.4f} | {item.tile_path.name}")


def plot_ranked_tiles(
    ranked_tiles: List[RankedTile],
    top_k: int,
    cols: int,
    save_figure: Optional[Path],
    show_figure: bool,
) -> None:
    if not ranked_tiles:
        return

    tiles_to_show = ranked_tiles if top_k == 0 else ranked_tiles[:top_k]
    n_tiles = len(tiles_to_show)
    cols = max(1, cols)
    rows = ceil(n_tiles / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for rank, (ax, item) in enumerate(zip(axes, tiles_to_show), start=1):
        ax.imshow(item.image, cmap="gray")
        ax.set_title(f"#{rank} | H={item.entropy:0.4f}")
        ax.set_xlabel(item.tile_path.name)
        ax.axis("off")

    for ax in axes[n_tiles:]:
        ax.axis("off")

    fig.suptitle("Tiles Ranked by Shannon Entropy", fontsize=14)
    fig.tight_layout()

    if save_figure is not None:
        save_figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_figure, dpi=200, bbox_inches="tight")
        print(f"\nSaved ranking figure to: {save_figure}")

    if show_figure:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    image_paths = iter_image_paths(input_path)

    print(f"Processing {len(image_paths)} tile image(s)...")
    all_ranked_tiles = rank_tiles(image_paths)
    if not all_ranked_tiles:
        raise SystemExit("No valid tiles were found for the provided input.")
    print_ranking(all_ranked_tiles)

    save_figure = (
        Path(args.save_figure).expanduser().resolve() if args.save_figure else None
    )
    plot_ranked_tiles(
        ranked_tiles=all_ranked_tiles,
        top_k=args.top_k,
        cols=args.cols,
        save_figure=save_figure,
        show_figure=not args.no_show,
    )


if __name__ == "__main__":
    main()
