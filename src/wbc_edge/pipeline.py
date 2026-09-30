import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4
import cv2
import numpy as np
from .runtime import RKNNModel
from .vision import letterbox, candidates, crop_cell, classify


class Pipeline:
    def __init__(self, cfg, model_factory=RKNNModel):
        self.cfg = cfg
        self.detector = model_factory(cfg["detector"]["model"])
        try:
            self.classifier = model_factory(cfg["classifier"]["model"])
        except Exception:
            self.detector.close()
            raise

    def close(self):
        try:
            self.detector.close()
        finally:
            self.classifier.close()

    def analyze(self, bgr):
        if bgr is None or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3 or not bgr.size:
            raise ValueError("Input image must be nonempty BGR uint8")
        start = perf_counter()
        d, c = self.cfg["detector"], self.cfg["classifier"]
        rgb, transform = letterbox(bgr, d["size"])
        boxes = candidates(self.detector.run(rgb), d, transform, bgr.shape)
        cells = []
        for box, score in boxes:
            crop, crop_box = crop_cell(bgr, box, d["crop_padding"])
            rgb = cv2.cvtColor(cv2.resize(crop, (c["size"], c["size"])), cv2.COLOR_BGR2RGB)
            label, confidence, probabilities = classify(self.classifier.run(rgb), c)
            cells.append({"box_xyxy": box.tolist(), "crop_xyxy": crop_box,
                          "detection_confidence": score, "label": label,
                          "classification_confidence": confidence,
                          "probabilities": probabilities,
                          "needs_review": confidence < c["review_threshold"]})
        return {"schema_version": 1, "sample_id": uuid4().hex,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "image_size": [bgr.shape[1], bgr.shape[0]], "class_labels": c["labels"],
                "review_threshold": c["review_threshold"], "cells": cells,
                "needs_review": any(x["needs_review"] for x in cells),
                "analysis_ms": round((perf_counter() - start) * 1000, 3)}


def annotate(bgr, result):
    image = bgr.copy()
    for cell in result["cells"]:
        x1, y1, x2, y2 = [round(v) for v in cell["box_xyxy"]]
        color = (0, 165, 255) if cell["needs_review"] else (60, 200, 60)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f'{cell["label"]} {cell["classification_confidence"]:.3f}'
        cv2.putText(image, text, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image


def save_sample(bgr, result, output_dir):
    folder = Path(output_dir) / result["sample_id"]
    folder.mkdir(parents=True, exist_ok=False)
    for name, image in (("image.jpg", bgr), ("annotated.jpg", annotate(bgr, result))):
        if not cv2.imwrite(str(folder / name), image):
            raise IOError(f"Could not save {name}")
    (folder / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return folder
