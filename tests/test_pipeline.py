"""CPU unit tests for tensor contracts and orchestration; no hardware inference."""
import json
import tempfile
import unittest
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch
import numpy as np
from wbc_edge.config import load_config
from wbc_edge.pipeline import Pipeline, save_sample
from wbc_edge.vision import letterbox, candidates, classify, decode
from wbc_edge.upload import upload_sample


ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT / "config/edge.json")
        self.cfg["detector"]["format"] = "decoded_xywh"
        self.image = np.zeros((100, 200, 3), dtype=np.uint8)

    def test_letterbox_roundtrip_and_nms(self):
        rgb, transform = letterbox(self.image, 640)
        self.assertEqual(rgb.shape, (640, 640, 3))
        outputs = [np.array([[[320, 320, 160, 160, .9, 1],
                              [320, 320, 160, 160, .7, 1]]], dtype=np.float32)]
        found = candidates(outputs, self.cfg["detector"], transform, self.image.shape)
        self.assertEqual(len(found), 1)
        np.testing.assert_allclose(found[0][0], [75, 25, 125, 75])

    def test_empty_detection(self):
        out = [np.zeros((1, 0, 6), np.float32)]
        self.assertEqual(candidates(out, self.cfg["detector"], (1, 1, 0, 0), self.image.shape), [])

    def test_raw_yolo_head_decoding(self):
        d = self.cfg["detector"].copy()
        d.update(format="yolov5_heads_nchw", anchors=[[[10, 20]]], size=640, head_logits=False)
        raw = np.array([.5, .5, .5, .5, .9, .8], np.float32).reshape(1, 6, 1, 1)
        np.testing.assert_allclose(decode([raw], d)[0], [320, 320, 10, 20, .9, .8])

    def test_bad_output_rejected(self):
        with self.assertRaises(ValueError):
            classify([np.array([1, 2, 3], np.float32)], self.cfg["classifier"])
        with self.assertRaises(ValueError):
            classify([np.array([np.nan] * 5, np.float32)], self.cfg["classifier"])
        with self.assertRaises(ValueError):
            classify([np.zeros(5, np.int8)], self.cfg["classifier"])

    def test_full_pipeline_and_persistence(self):
        closed = []
        class TensorFixture:
            def __init__(self, path):
                self.detection = "yolov5" in str(path)
            def run(self, rgb):
                if self.detection:
                    return [np.array([[[320, 320, 160, 160, .9, 1]]], np.float32)]
                return [np.zeros((1, 5), np.float32)]
            def close(self):
                closed.append(self.detection)
        pipeline = Pipeline(self.cfg, model_factory=TensorFixture)
        try:
            result = pipeline.analyze(self.image)
        finally:
            pipeline.close()
        self.assertEqual(len(closed), 2)
        self.assertTrue(result["needs_review"])
        self.assertEqual(result["cells"][0]["label"], "basophil")
        with tempfile.TemporaryDirectory() as tmp:
            folder = save_sample(self.image, result, tmp)
            self.assertTrue((folder / "image.jpg").is_file())
            self.assertEqual(json.loads((folder / "result.json").read_text())["sample_id"], result["sample_id"])

    def test_classifier_init_failure_releases_detector(self):
        closed = []
        class InitFixture:
            def __init__(self, path):
                if "classifier" in str(path):
                    raise RuntimeError("initialization failed")
            def close(self):
                closed.append(True)
        with self.assertRaises(RuntimeError):
            Pipeline(self.cfg, model_factory=InitFixture)
        self.assertEqual(closed, [True])

    def test_missing_model_reports_error(self):
        from wbc_edge.runtime import RKNNModel
        with self.assertRaises(FileNotFoundError):
            RKNNModel(ROOT / "models/nonexistent.rknn")

    def test_probability_contract(self):
        c = self.cfg["classifier"].copy()
        c["output"] = "probabilities"
        with self.assertRaises(ValueError):
            classify([np.ones(5, np.float32)], c)
        label, score, _ = classify([np.array([0, 0, 1, 0, 0], np.float32)], c)
        self.assertEqual((label, score), ("lymphocyte", 1))

    def test_http_upload_contract_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "image.jpg").write_bytes(b"image-test-fixture")
            (folder / "result.json").write_text('{"sample_id":"case-1"}')
            cfg = self.cfg["upload"].copy()
            cfg["url"] = "http://127.0.0.1:8000/upload"
            with patch("wbc_edge.upload.requests.post") as post:
                post.return_value.status_code = 200
                self.assertEqual(upload_sample(folder, cfg), 200)
                self.assertEqual(set(post.call_args.kwargs["files"]), {"image", "result"})
                self.assertEqual(post.call_args.kwargs["headers"]["Idempotency-Key"], "case-1")
                post.return_value.status_code = 503
                with self.assertRaises(RuntimeError):
                    upload_sample(folder, cfg)
            self.assertTrue((folder / "result.json").exists())

    def test_upload_to_local_http_receiver(self):
        received = {}
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received["body"] = self.rfile.read(int(self.headers["Content-Length"]))
                received["key"] = self.headers.get("Idempotency-Key")
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"accepted")
            def log_message(self, *args):
                pass
        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                (folder / "image.jpg").write_bytes(b"test-image-payload")
                (folder / "result.json").write_text('{"sample_id":"network-case"}')
                cfg = self.cfg["upload"].copy()
                cfg["url"] = f"http://127.0.0.1:{server.server_port}/upload"
                self.assertEqual(upload_sample(folder, cfg), 200)
            self.assertEqual(received["key"], "network-case")
            self.assertIn(b'name="image"', received["body"])
            self.assertIn(b'name="result"', received["body"])
            self.assertIn(b"test-image-payload", received["body"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
