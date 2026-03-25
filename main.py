import argparse
from pathlib import Path
from typing import Dict
from file_io.flatten_and_index import find_repo_root, create_update_db
from preprocessing.tiling import process_images
from bounding_boxes.parasite_detector import run_detection
from bounding_boxes.review_tool import run_review_tool

# Main parser
ap = argparse.ArgumentParser(description="Parasite image processing pipeline.")
ap.add_argument("--repo-root", type=str, default=None, help="Optional override for repo root.")

# Subparsers for different commands
subparsers = ap.add_subparsers(dest='command', required=True, help='Action to run')

# --- 'db' command ---
parser_db = subparsers.add_parser('db', help='Create or update the file index database.')
parser_db.add_argument("--rehash-all", action="store_true", help="Hash all files every run (slower, safest).")
parser_db.add_argument("--keep-duplicates", action="store_true", help="Allow duplicate hashes to be copied/indexed.")

# --- 'tile' command ---
parser_tile = subparsers.add_parser('tile', help='Extract tiles from microscope images.')
parser_tile.add_argument("--source-dir", type=str, default="./data/preprocess/flat_images", help="Source directory for images to be tiled.")
parser_tile.add_argument("--dest-dir", type=str, default="./data/preprocess/tiles", help="Destination directory for created tiles.")
parser_tile.add_argument("--microscope-type", 
                         choices=["Carson_Microscope", "Foldscope", "Lab_Microscope", "Uhandy_Microscope"],
                         help="Filter images by microscope type (from filename).")
parser_tile.add_argument("--parasite-type", 
                         choices=["Ascaris_Lumbricoides", "Balantidium_Coli", "Clonorchis_Sinensis", 
                                  "Diphyllibothrium_Latum", "Dipylidium_Cannum", 
                                  "Echinococcus_Gran", "Endolimax_Nana", 
                                  "Entamoeba_Coli", "Entamoeba_Histolytica", 
                                  "Enterobius_Vermicularis", "Fasciola_Hepatica", "Giardia", 
                                  "Hymenolepis_Nana", "Necator_Americanus", "Schistosoma_Mansoni",
                                  "Taenia_Pisiformis", "Taenia_Saginata", "Taenia_Solium", 
                                  "Trichuria_Trichuris"],
                         default="Entamoeba_Coli",
                         help="Filter images by parasite type (from filename).")
parser_tile.add_argument("--patch-size", type=int, default=1024, help="Size of square tiles to extract.")
parser_tile.add_argument("--stride", type=int, default=256, help="Stride for tile extraction.")
parser_tile.add_argument("--min-fov-fraction", type=float, default=1., help="Min fraction of tile in FOV.")

# --- 'detect' command ---
parser_detect = subparsers.add_parser('detect', help='Run automated bounding box proposal detection.')
parser_detect.add_argument("input_dir", help="Directory containing microscope images")
parser_detect.add_argument("--output-dir", default="./bounding_boxes/detection_output", help="Directory to save detection results.")
parser_detect.add_argument("--min-frac", type=float, default=0.0003)
parser_detect.add_argument("--max-frac", type=float, default=0.20)
parser_detect.add_argument("--nms-iou", type=float, default=0.3)
parser_detect.add_argument("--format", choices=["coco","yolo","both"], default="both")
parser_detect.add_argument("--no-vis", action="store_true", help="Skip visualization output")

# --- 'review' command ---
parser_review = subparsers.add_parser('review', help='Launch the annotation review web tool.')
parser_review.add_argument("--images", default='./data/preprocess/flat_images', help="Directory containing microscope images")
parser_review.add_argument("--proposals", default='./bounding_boxes/detection_output/proposals.json', help="proposals.json from parasite_detector.py")
parser_review.add_argument("--annotations", default=None, help="Existing annotations.json to resume (optional)")
parser_review.add_argument("--output", default="./bounding_boxes/annotations.json", help="Where to auto-save annotations (default: ./bounding_boxes/annotations.json)")
parser_review.add_argument("--port", type=int, default=5000)
parser_review.add_argument("--host", default="127.0.0.1")
parser_review.add_argument("--no-browser", action="store_true")


args = ap.parse_args()

def run_create_update_db(args): 
    repo_root = Path(args.repo_root).expanduser().resolve() if args.repo_root else find_repo_root(Path.cwd())
    create_update_db(repo_root, args.rehash_all, args.keep_duplicates)
    
def run_tiling(args):
    source_path = Path(args.source_dir).expanduser().resolve()
    dest_path = Path(args.dest_dir).expanduser().resolve()
    
    process_images(
        source_dir=source_path,
        dest_dir=dest_path,
        microscope_type=args.microscope_type,
        parasite_type=args.parasite_type,
        patch_size=args.patch_size,
        stride=args.stride,
        min_fov_fraction=args.min_fov_fraction
    )

# Main execution logic
if args.command == 'db':
    run_create_update_db(args)
elif args.command == 'tile':
    run_tiling(args)
elif args.command == 'detect':
    run_detection(args)
elif args.command == 'review':
    run_review_tool(args)
else:
    print(f"Error: Unknown command '{args.command}'")
    ap.print_help()