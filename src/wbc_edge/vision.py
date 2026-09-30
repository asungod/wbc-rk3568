import cv2
import numpy as np


def float_tensor(value):
    a = np.asarray(value)
    if not np.issubdtype(a.dtype, np.floating) or not np.isfinite(a).all():
        raise ValueError("Outputs must be finite, dequantized floating-point tensors")
    return a.astype(np.float32, copy=False)


def letterbox(bgr, size):
    h, w = bgr.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    left, top = (size - nw) // 2, (size - nh) // 2
    rgb = cv2.cvtColor(cv2.resize(bgr, (nw, nh)), cv2.COLOR_BGR2RGB)
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = rgb
    # Use actual rounded resize ratios to undo the transform.
    return canvas, (nw / w, nh / h, left, top)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80, 80)))


def decode(outputs, cfg):
    nc = cfg["classes"]
    if cfg["format"] == "decoded_xywh":
        if len(outputs) != 1:
            raise ValueError("decoded_xywh expects one output")
        x = float_tensor(outputs[0])
        if x.ndim == 3 and x.shape[0] == 1:
            x = x[0]
        if x.ndim != 2 or x.shape[1] != 5 + nc:
            raise ValueError(f"Expected [N,{5 + nc}] output, got {x.shape}")
        return x
    if cfg["format"] != "yolov5_heads_nchw":
        raise ValueError("Unsupported detector output format")
    anchors = cfg["anchors"]
    if len(outputs) != len(anchors):
        raise ValueError("Head count does not match anchor groups")
    decoded = []
    for output, group in zip(outputs, anchors):
        raw = float_tensor(output)
        na = len(group)
        if raw.ndim != 4 or raw.shape[:2] != (1, na * (5 + nc)):
            raise ValueError(f"Expected NCHW YOLOv5 head, got {raw.shape}")
        gh, gw = raw.shape[2:]
        x = raw.reshape(na, 5 + nc, gh, gw).transpose(0, 2, 3, 1)
        if cfg["head_logits"]:
            x = sigmoid(x)
        elif np.any((x < 0) | (x > 1)):
            raise ValueError("Activated YOLOv5 heads must be in [0,1]")
        yy, xx = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
        grid = np.stack((xx, yy), axis=-1)[None]
        xy = (x[..., :2] * 2 - 0.5 + grid) * [cfg["size"] / gw, cfg["size"] / gh]
        wh = (x[..., 2:4] * 2) ** 2 * np.asarray(group)[:, None, None, :]
        decoded.append(np.concatenate((xy, wh, x[..., 4:]), axis=-1).reshape(-1, 5 + nc))
    return np.concatenate(decoded)


def nms(boxes, scores, threshold):
    order = np.argsort(-scores, kind="stable")
    keep = []
    area = np.prod(np.maximum(0, boxes[:, 2:] - boxes[:, :2]), axis=1)
    while len(order):
        i = int(order[0])
        keep.append(i)
        others = order[1:]
        inter = np.prod(np.maximum(0, np.minimum(boxes[i, 2:], boxes[others, 2:]) -
                                   np.maximum(boxes[i, :2], boxes[others, :2])), axis=1)
        union = area[i] + area[others] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = others[iou <= threshold]
    return keep


def candidates(outputs, cfg, transform, shape):
    x = decode(outputs, cfg)
    if np.any(x[:, 2:4] < 0) or np.any((x[:, 4:] < 0) | (x[:, 4:] > 1)):
        raise ValueError("Decoded sizes/probabilities are invalid")
    class_id = np.argmax(x[:, 5:], axis=1)
    score = x[:, 4] * np.max(x[:, 5:], axis=1)
    mask = (score >= cfg["confidence"]) & (class_id == cfg["wbc_class_id"])
    x, score = x[mask], score[mask]
    box = np.concatenate((x[:, :2] - x[:, 2:4] / 2, x[:, :2] + x[:, 2:4] / 2), axis=1)
    sx, sy, left, top = transform
    box = (box - [left, top, left, top]) / [sx, sy, sx, sy]
    h, w = shape[:2]
    box = np.clip(box, [0, 0, 0, 0], [w, h, w, h])
    valid = np.all(box[:, 2:] - box[:, :2] >= cfg["min_side"], axis=1)
    box, score = box[valid], score[valid]
    keep = nms(box, score, cfg["iou"])[:cfg["max_candidates"]]
    return [(box[i], float(score[i])) for i in keep]


def crop_cell(bgr, box, padding):
    h, w = bgr.shape[:2]
    margin = (box[2:] - box[:2]) * padding
    lo = np.maximum(0, np.floor(box[:2] - margin)).astype(int)
    hi = np.minimum([w, h], np.ceil(box[2:] + margin)).astype(int)
    crop = bgr[lo[1]:hi[1], lo[0]:hi[0]]
    if not crop.size:
        raise ValueError("Empty cell crop")
    return crop.copy(), [int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])]


def classify(outputs, cfg):
    if len(outputs) != 1:
        raise ValueError("Classifier must expose one five-class output")
    p = float_tensor(outputs[0]).reshape(-1)
    if len(p) != len(cfg["labels"]):
        raise ValueError("Class output length does not match labels")
    if cfg["output"] == "logits":
        p = np.exp(p - p.max())
        p = p / p.sum()
    elif cfg["output"] == "probabilities":
        if np.any((p < 0) | (p > 1)) or not np.isclose(p.sum(), 1, atol=1e-3):
            raise ValueError("Classifier probabilities must sum to one")
    else:
        raise ValueError("Unsupported classifier output")
    i = int(np.argmax(p))
    return cfg["labels"][i], float(p[i]), p.tolist()
