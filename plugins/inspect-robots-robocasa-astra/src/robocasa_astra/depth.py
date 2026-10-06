"""Transport metric depth without quantization and derive explicitly scaled previews."""

import base64
import io
import zlib

import numpy as np

PREVIEW_MAX_METERS = 3.0


def normalize_depth_buffer(buffer):
    """Accept renderer HxW or HxWx1 depth and reject corrupted normalized values."""
    values = np.asarray(buffer)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Invalid normalized renderer depth")
    if np.any(values < -1e-6) or np.any(values > 1 + 1e-6):
        raise ValueError("Renderer depth must be normalized to [0, 1]")
    return np.clip(values, 0, 1)


def validate_depth(depth, shape=None):
    """Require finite positive optical-axis distances aligned to an RGB image."""
    values = np.asarray(depth, dtype=np.float32)
    if values.ndim != 2 or (shape is not None and values.shape != shape):
        raise ValueError("Depth must match the RGB height and width")
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Depth must contain finite positive meters")
    return values


def encode_depth(depth):
    """Send lossless float32 NPY data through compressed JSON-line transport."""
    buffer = io.BytesIO()
    np.save(buffer, validate_depth(depth), allow_pickle=False)
    return base64.b64encode(zlib.compress(buffer.getvalue())).decode("ascii")


def decode_depth(payload, shape):
    """Decode only bounded, non-pickle float32 camera arrays."""
    decoder = zlib.decompressobj()
    maximum = shape[0] * shape[1] * 4 + 4096
    data = decoder.decompress(base64.b64decode(payload, validate=True), maximum + 1)
    if len(data) > maximum or not decoder.eof or decoder.unused_data:
        raise ValueError("Invalid or oversized depth payload")
    values = np.load(io.BytesIO(data), allow_pickle=False)
    if values.dtype != np.float32:
        raise ValueError("Depth transport requires float32")
    return validate_depth(values, shape)


def preview_depth(depth):
    """Map 0..3 meters to white..black; saturation never changes saved metric data."""
    values = validate_depth(depth)
    grey = np.rint(255 * (1 - np.clip(values / PREVIEW_MAX_METERS, 0, 1))).astype(np.uint8)
    return np.repeat(grey[..., None], 3, axis=2)


def depth_summary(depth):
    """Provide sampled metric distances with their exact pixel coordinates."""
    values = validate_depth(depth)
    ys = np.linspace(0, values.shape[0] - 1, 8).astype(int)
    xs = np.linspace(0, values.shape[1] - 1, 8).astype(int)
    return {
        "unit": "m",
        "quantity": "camera optical-axis depth, not Euclidean range",
        "pixel_origin": "top-left",
        "preview": {"near_white_m": 0, "far_black_m": PREVIEW_MAX_METERS},
        "min_m": float(values.min()),
        "max_m": float(values.max()),
        "sample_x_pixels": xs.tolist(),
        "sample_y_pixels": ys.tolist(),
        "sample_depth_m": np.round(values[np.ix_(ys, xs)], 4).tolist(),
    }
