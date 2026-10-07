"""Condition-specific depth inputs and query schemas, isolated from action execution."""

from __future__ import annotations

import base64
import copy
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..depth import normalize_depth_buffer, preview_depth, validate_depth
from ..depth_query import query_depth
from .prompts import RESPONSE_SCHEMA, image_part, text_part

CONDITIONS = ("rgb", "color", "pixel", "grid", "hybrid")
QUERY_ROUNDS = 4


def response_schema(condition: str) -> dict:
    """Queries and actions are mutually exclusive; hybrid enables images and pixel queries."""
    schema = copy.deepcopy(RESPONSE_SCHEMA)
    query_kind = "pixel" if condition == "hybrid" else condition
    if condition in ("pixel", "grid", "hybrid"):
        fields = {
            "observation_id": {"type": "string"},
            "camera": {"type": "string"},
            "kind": {"type": "string", "enum": [query_kind]},
        }
        if query_kind == "pixel":
            fields.update(
                {
                    "u": {"type": "integer"},
                    "v": {"type": "integer"},
                    "radius": {"type": "integer", "minimum": 0, "maximum": 4},
                }
            )
        else:
            fields["cell_id"] = {"type": "string", "pattern": "^r(0[0-9]|1[0-5])c(0[0-9]|1[0-5])$"}
        schema["properties"]["queries"] = {
            "type": "array",
            "minItems": 1,
            "maxItems": 32,
            "items": {
                "type": "object",
                "properties": fields,
                "required": list(fields),
                "additionalProperties": False,
            },
        }
        schema["properties"]["actions"] = {
            "anyOf": [schema["properties"]["actions"], {"type": "null"}]
        }
        schema["properties"]["queries"] = {
            "anyOf": [schema["properties"]["queries"], {"type": "null"}]
        }
        schema["required"].append("queries")
    return schema


def instructions(condition: str) -> str:
    """Describe only the enabled condition, without exposing depth to RGB-only requests."""
    if condition == "rgb":
        return ""
    image_note = (
        "DEPTH IMAGES: aligned grayscale camera Z; near is white, far is "
        "black. Each image has its own metre scale."
    )
    if condition == "color":
        return image_note
    kind = "pixel" if condition == "hybrid" else condition
    query_note = (
        f"DEPTH QUERY: instead of actions, return queries (kind={kind}), "
        "using the current observation_id and camera name. Coordinates (u,v) "
        "use top-left origin and the matching RGB resolution; Z is optical-axis "
        "distance in metres, not Euclidean range. "
    )
    query_note += (
        "Pixel queries require u,v,radius (0..4). "
        if kind == "pixel"
        else "Grid queries require cell_id r00c00..r15c15 on a 16x16 partition. "
    )
    query_note += (
        "At most four query rounds per observation, then return actions. "
        "A query does not advance physics. Always include both fields: for a query set actions=null; for actions set queries=null. Exactly one field must be non-null."
    )
    if condition == "hybrid":
        return (
            image_note
            + "\n\n"
            + query_note
            + (
                " Both inputs are available together: use the depth images for spatial "
                "context and optionally query precise pixel Z from the same observation. "
                "A pixel query does not replace or remove the depth images."
            )
        )
    return query_note


def render_depths(env, views: list[dict]) -> dict[str, np.ndarray]:
    """Render metric Z at each RGB camera's exact resolution and orientation."""
    from robosuite.utils.camera_utils import get_real_depth_map

    result = {}
    for view in views:
        with Image.open(io.BytesIO(base64.b64decode(view["png"]))) as image:
            width, height = image.size
        _, buffer = env.sim.render(width=width, height=height, camera_name=view["name"], depth=True)
        result[view["name"]] = validate_depth(
            get_real_depth_map(env.sim, normalize_depth_buffer(buffer))[::-1], (height, width)
        )
    return result


def attach_previews(reply: dict, depths: dict[str, np.ndarray]) -> None:
    """Attach depth previews only to the depth-image condition's simulator response."""
    reply["depth_views"] = []
    for camera, depth in depths.items():
        buffer = io.BytesIO()
        Image.fromarray(preview_depth(depth)).save(buffer, format="PNG")
        reply["depth_views"].append(
            {
                "name": camera,
                "png": base64.b64encode(buffer.getvalue()).decode(),
                "scale_m": [float(np.percentile(depth, 2)), float(np.percentile(depth, 98))],
            }
        )


def extra_parts(obs: dict, folder: Path, condition: str) -> list[dict]:
    """Persist input images and bind all depth input to this observation."""
    if condition == "rgb":
        return []
    parts = [
        text_part(
            f"observation_id={obs['observation_id']}; cameras="
            + json.dumps([v["name"] for v in obs["views"]])
        )
    ]
    if condition in ("color", "hybrid"):
        folder.mkdir(parents=True, exist_ok=True)
        for view in obs["depth_views"]:
            path = folder / f"{view['name']}-depth.png"
            path.write_bytes(base64.b64decode(view["png"]))
            parts += [
                text_part(f"Depth camera={view['name']}; near/far scale_m={view['scale_m']}"),
                image_part(path),
            ]
    return parts


def answer_queries(depths: dict, requests: list, observation_id: str, condition: str) -> list[dict]:
    """Validate every query before returning any answers or advancing an action."""
    if condition not in ("pixel", "grid", "hybrid") or not 1 <= len(requests) <= 32:
        raise ValueError("depth query unavailable or invalid batch size")
    kind = "pixel" if condition == "hybrid" else condition
    return [query_depth(depths, request, observation_id, {kind}) for request in requests]
