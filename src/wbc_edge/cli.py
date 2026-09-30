import argparse
import json
import sys
import cv2
from .config import load_config
from .pipeline import Pipeline, save_sample
from .upload import upload_sample


def main():
    parser = argparse.ArgumentParser(description="RK3568 WBC image analysis")
    parser.add_argument("--config", default="config/edge.json")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image")
    source.add_argument("--camera", type=int, help="Capture one camera frame")
    parser.add_argument("--upload", action="store_true", help="Upload this sample when it needs review")
    args = parser.parse_args()
    try:
        cfg = load_config(args.config)
        if args.image:
            image = cv2.imread(args.image)
        else:
            cap = cv2.VideoCapture(args.camera)
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera"]["width"])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["height"])
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError("Camera frame capture failed")
            finally:
                cap.release()
        pipe = Pipeline(cfg)
        try:
            result = pipe.analyze(image)
        finally:
            pipe.close()
        folder = save_sample(image, result, cfg["output_dir"])
        print(json.dumps({"sample_dir": str(folder), "result": result}, ensure_ascii=False))
        if args.upload and result["needs_review"]:
            print(f"Upload HTTP {upload_sample(folder, cfg['upload'])}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
