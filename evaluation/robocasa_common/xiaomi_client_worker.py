"""Use the installed official Xiaomi client inside a private Spark GPU container."""

import base64
import contextlib
import io
import json
import sys
import time

import numpy as np
from PIL import Image


def controller_compatible_actions(actions):
    """Match native continuous input saturation; preserve both discrete thresholds."""
    original = np.asarray(actions)
    if original.ndim != 2 or original.shape[1] != 12 or not np.isfinite(original).all():
        raise ValueError("Invalid finite official 12D action chunk")
    applied = original.copy()
    continuous = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10]
    applied[:, continuous] = np.clip(applied[:, continuous], -1, 1)
    return applied


def main():
    """Keep stdout reserved for JSON; persist no frames or videos on Spark."""
    sys.path.insert(0, "/work")
    with contextlib.redirect_stdout(sys.stderr):
        from rollout import EvalClient

        client = EvalClient("/checkpoint", "localhost", 10086, "robocasa365", 0.95)
        client.client.client_socket.settimeout(150)
    try:
        for line in sys.stdin:
            started = time.monotonic()
            try:
                request = json.loads(line)
                images = {
                    key: np.stack(
                        [
                            np.asarray(Image.open(io.BytesIO(base64.b64decode(v))).convert("RGB"))
                            for v in values
                        ]
                    )
                    for key, values in request["images"].items()
                }
                with contextlib.redirect_stdout(sys.stderr):
                    actions = client.infer(
                        np.asarray(request["state"], dtype=np.float32),
                        images,
                        request["instruction"],
                    )
                applied = controller_compatible_actions(actions)
                response = {
                    "actions": applied.tolist(),
                    "original_actions": actions.tolist(),
                    "controller_profile": "native-continuous-input-saturation-v1",
                    "saturated_values": int(np.count_nonzero(applied != actions)),
                    "seconds": time.monotonic() - started,
                }
            except Exception as error:
                response = {"error": repr(error)}
            print(json.dumps(response), flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
