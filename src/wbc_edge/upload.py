import json
import os
from pathlib import Path
from urllib.parse import urlparse
import requests


def upload_sample(folder, cfg):
    """Send image + result.json once; failures leave local files intact."""
    folder = Path(folder)
    url = cfg["url"]
    parsed = urlparse(url)
    if parsed.scheme not in ("https", "http") or not parsed.hostname:
        raise ValueError("Configure an HTTP(S) upload URL")
    if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("Use HTTPS for non-local uploads")
    paths = [folder / "image.jpg", folder / "result.json"]
    if sum(p.stat().st_size for p in paths) > cfg["max_bytes"]:
        raise ValueError("Sample exceeds upload size limit")
    result = json.loads(paths[1].read_text(encoding="utf-8"))
    headers = {"Idempotency-Key": result["sample_id"]}
    token = os.environ.get(cfg["token_env"], "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with paths[0].open("rb") as image, paths[1].open("rb") as metadata:
        response = requests.post(url, headers=headers,
                                 files={"image": ("image.jpg", image, "image/jpeg"),
                                        "result": ("result.json", metadata, "application/json")},
                                 timeout=(5, cfg["timeout"]), allow_redirects=False)
    if not 200 <= response.status_code < 300:
        raise RuntimeError(f"Upload rejected: HTTP {response.status_code}")
    return response.status_code
