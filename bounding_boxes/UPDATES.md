3/23/2026

# parasite_detector.py

On launch, checks if proposals.json already exists in the output dir
Builds a set of already-processed filenames from it
Only runs detection on new images not already in that set
Merges new results into the existing COCO JSON (correct id sequencing)
Prints skip/new counts so collaborators can see what happened
Pass --force (via argparse, which you'll want to add) or delete proposals.json to re-run everything

# review_tool.py

export_annotations() now outputs COCO format (same structure as proposals.json) with status and label added per annotation. So annotations.json is directly loadable back into the app as proposals.
load_existing_annotations() auto-detects whether it's reading the old dict-keyed format or the new COCO list format, so existing annotation files still work.