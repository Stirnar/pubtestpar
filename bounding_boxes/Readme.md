#Dependancies:

pip install flask opencv-python-headless numpy scikit-image

#Quick Start:

Step 1:
python3 parasite_detector.py ./images --output-dir ./output

Step 2:
python3 review_tool.py --images ./images --proposals ./output/proposals.json

	Can also resume a previous session
python3 review_tool.py --images ./images --proposals ./output/proposals.json \
                       --annotations ./annotations.json


#parasite_detector uses:
Stained specimen detection: HSV color segmentation for pink, purple, brown, red, blue stains
Local contrast detection: Background subtraction at multiple sensitivity levels
Edge density detection: Finds textured objects against smooth backgrounds
Dark object detection: Intensity thresholding for objects darker than background
Local anomaly detection: Z-score based detection for faint/subtle objects

Sensitivity modulation:

More proposals (higher recall, more false positives)
python3 parasite_detector.py ./images --min-frac 0.0001

Fewer proposals (lower recall, fewer false positives)
python3 parasite_detector.py ./images --min-frac 0.001

Default is 0.0003

#Review tool.py opens a browser GUI to loade you parasite_detector.py images where you can:

Review each bounding box: approve, reject, or add your own
Label each box with species names
Export approved annotations as COCO JSON or YOLO format

Loads images + proposals.json from parasite_detector.py.
Human reviews each bounding box: approve, reject, relabel, or draw new ones.
Exports annotations.json keyed by filename.

Usage:
    After running parasite_detector.py:
    python3 review_tool.py --images ./images --proposals ./detection_output/proposals.json

    Resume a previous review session:
    python3 review_tool.py --images ./images --proposals ./detection_output/proposals.json \
                           --annotations ./annotations.json

    Run on a different port:
    python3 review_tool.py --images ./images --proposals ./proposals.json --port 8080

	output: annotations.json
    {
      "version": "1.0",
      "images": {
        "Giardia_lamblia.jpg": {
          "width": 952, "height": 1288,
          "annotations": [
            {"bbox": [x, y, w, h], "label": "Giardia lamblia", "status": "approved"},
            {"bbox": [x, y, w, h], "label": "noise", "status": "approved"},
            {"bbox": [x, y, w, h], "label": "", "status": "rejected"},
          ]
        }
      }
    }

#annotations.json format:

```json
{
  "version": "1.0",
  "generator": "review_tool.py",
  "images": {
    "Giardia_lamblia.jpg": {
      "width": 952,
      "height": 1288,
      "annotations": [
        {
          "bbox": [747, 632, 38, 33],
          "label": "Giardia lamblia",
          "status": "approved",
          "method": "purple_stain"
        },
        {
          "bbox": [450, 516, 64, 63],
          "label": "noise",
          "status": "approved",
          "method": "dark_object"
        },
        {
          "bbox": [300, 200, 30, 28],
          "label": "",
          "status": "rejected",
          "method": "anomaly_k61_z2.0"
        }
      ]
    }
  }
}
```

#Keyboard Shortcuts (thanks claude)
Key

A
Approve current box

R
Reject current box

D
Draw a new box manually

← →
Navigate between images

↑ ↓
Navigate between boxes

Tab
Jump to next pending box