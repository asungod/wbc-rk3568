import json
import math
from pathlib import Path


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    d, c = cfg["detector"], cfg["classifier"]
    for value in (d["size"], c["size"], d["classes"], d["max_candidates"]):
        if type(value) is not int or value <= 0:
            raise ValueError("Sizes, class count and candidate limit must be positive integers")
    if not 0 <= d["wbc_class_id"] < d["classes"]:
        raise ValueError("Invalid detector WBC class id")
    for value in (d["confidence"], d["iou"], c["review_threshold"]):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Thresholds must be in [0,1]")
    for value in (d["min_side"], d["crop_padding"]):
        if not math.isfinite(value) or value < 0:
            raise ValueError("Crop parameters must be finite and nonnegative")
    labels = c["labels"]
    if len(labels) != 5 or len(set(labels)) != 5 or not all(isinstance(x, str) and x for x in labels):
        raise ValueError("Supply five distinct classifier labels in model output order")
    if d["format"] not in ("yolov5_heads_nchw", "decoded_xywh"):
        raise ValueError("Unsupported detector format")
    if c["output"] not in ("logits", "probabilities"):
        raise ValueError("Unsupported classifier output")
    if d["format"] == "yolov5_heads_nchw":
        for group in d["anchors"]:
            if not group or not all(len(a) == 2 and all(math.isfinite(v) and v > 0 for v in a) for a in group):
                raise ValueError("Anchors must be positive width/height pairs")
    u = cfg["upload"]
    if not math.isfinite(u["timeout"]) or u["timeout"] <= 0 or u["max_bytes"] <= 0:
        raise ValueError("Upload limits must be positive")
    for section in (d, c):
        section["model"] = str((path.parent / section["model"]).resolve())
    cfg["output_dir"] = str((path.parent / cfg["output_dir"]).resolve())
    return cfg
