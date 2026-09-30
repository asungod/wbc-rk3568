from pathlib import Path
import numpy as np


class RKNNModel:
    """Own one RKNNLite runtime; inference is serialized by its caller."""

    def __init__(self, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"RKNN model not found: {path}")
        from rknnlite.api import RKNNLite
        self._runtime = RKNNLite()
        try:
            ret = self._runtime.load_rknn(str(path))
            if ret != 0:
                raise RuntimeError(f"load_rknn failed ({ret}): {path}")
            # RK3568: use the runtime default instead of RK3588 core masks.
            ret = self._runtime.init_runtime()
            if ret != 0:
                raise RuntimeError(f"init_runtime failed ({ret}): {path}")
        except Exception:
            self.close()
            raise

    def run(self, rgb):
        if self._runtime is None:
            raise RuntimeError("RKNN runtime already released")
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Input must be HWC RGB uint8")
        outputs = self._runtime.inference(
            inputs=[np.ascontiguousarray(rgb[None, ...])], data_format=["nhwc"]
        )
        if outputs is None or len(outputs) == 0:
            raise RuntimeError("RKNN inference returned no outputs")
        return outputs

    def close(self):
        runtime = getattr(self, "_runtime", None)
        self._runtime = None
        if runtime is not None:
            runtime.release()
