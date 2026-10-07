"""Read calibrated camera Z with explicit pixel conventions and observation identity."""

import re

import numpy as np

from robocasa_astra.depth import validate_depth


def statistics(values):
    """Describe a region rather than treating its center pixel as its whole surface."""
    a = validate_depth(values)
    return {
        "count": int(a.size),
        "min_m": float(a.min()),
        "max_m": float(a.max()),
        "median_m": float(np.median(a)),
        "mean_m": float(a.mean()),
        "p10_m": float(np.percentile(a, 10)),
        "p90_m": float(np.percentile(a, 90)),
    }


def query_depth(depths, request, observation_id, enabled):
    """Reject stale, invalid or disabled requests without advancing simulation."""
    if request["observation_id"] != observation_id:
        raise ValueError("Stale observation_id")
    camera = request["camera"]
    if camera not in depths:
        raise ValueError("Unknown camera")
    kind = request["kind"]
    if kind not in enabled:
        raise ValueError("Query kind is unavailable in this condition")
    values = validate_depth(depths[camera])
    h, w = values.shape
    if kind == "pixel":
        u, v, radius = request["u"], request["v"], request["radius"]
        if any(type(x) is not int for x in (u, v, radius)):
            raise ValueError("Pixel coordinates must be integers")
        if not (0 <= u < w and 0 <= v < h and 0 <= radius <= 4):
            raise ValueError("Pixel or neighborhood out of bounds")
        x0, x1 = max(0, u - radius), min(w, u + radius + 1)
        y0, y1 = max(0, v - radius), min(h, v + radius + 1)
    elif kind == "grid":
        match = re.fullmatch(r"r(\d{2})c(\d{2})", request["cell_id"])
        if match is None:
            raise ValueError("Cell ID must be r00c00 through r15c15")
        r, c = map(int, match.groups())
        if not (0 <= r < 16 and 0 <= c < 16):
            raise ValueError("Cell out of bounds")
        xs, ys = np.linspace(0, w, 17, dtype=int), np.linspace(0, h, 17, dtype=int)
        x0, x1, y0, y1 = int(xs[c]), int(xs[c + 1]), int(ys[r]), int(ys[r + 1])
        u, v = (x0 + x1 - 1) // 2, (y0 + y1 - 1) // 2
    else:
        raise ValueError("Unknown query kind")
    return {
        "camera": camera,
        "observation_id": observation_id,
        "kind": kind,
        "unit": "m",
        "quantity": "camera optical-axis Z, not Euclidean range",
        "pixel_origin": "top-left",
        "pixel_uv": [u, v],
        "pixel_depth_m": float(values[v, u]),
        "region_bounds_uv_half_open": [x0, y0, x1, y1],
        "region_statistics": statistics(values[y0:y1, x0:x1]),
    }
