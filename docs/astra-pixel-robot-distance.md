# Pixel-to-robot distance queries

Branch: codex/vla-pixel-robot-distance. Enable with `--condition spatial`.
This condition uses hybrid RGB/depth inputs plus pixel Z and spatial queries. Existing conditions are unchanged.

## Request

```json
{"observation_id":"<current observation_id>","camera":"robot0_eye_in_hand","kind":"spatial","u":116,"v":184,"radius":0,"robot_part":"gripper"}
```

Return this in the existing `queries` array, with `actions=null`. The other response fields retain the normal schema. Use `queries=null` for actions. Each observation allows four query rounds and batches of 1 to 32 queries. To request only Z, use `kind=pixel` and `robot_part=null`.

## Robot parts and coordinates

The prompt lists available robot parts. `gripper` refers to the existing controller grip site between fingertips. `base` refers to the native base-center site. Named robot bodies and sites are also available as `body:<name>` and `site:<name>`; their origins are not necessarily their surface or physical finger endpoints. Scene object names are not exposed.

Each answer includes `distance_m`, `point_world_m`, `robot_part_world_m`, and displacement vectors `delta_world_m`, `delta_part_local_m`, `delta_base_local_m`. Vectors point FROM the requested robot origin TO the image surface. Base-local axes are forward, left, up. Part-local axes use the native body/site orientation. For the gripper, x is the finger closing axis and z is the approach axis. Units are metres.

The renderer's metric optical-axis Z is unprojected with the camera intrinsics and world pose captured at the same observation. A selected pixel measures its visible surface, including background; it does not reveal an occluded target. Empty far-plane pixels, stale observations, invalid pixels and unavailable robot parts are rejected. A measured straight-line distance is not a reachability or collision-free path guarantee.

## Persistence and operation

Queries do not advance physics. Camera calibration and robot frames are frozen, included in observation identity and saved to `worker/geometry/<observation_id>.json`. Questions and answers use the existing `policy/progress.json` and `trace.jsonl` query history, including resume and per-call token/latency accounting. The existing no-fixed-model-timeout runner and native checkpoint recovery remain in place.

No experiments are launched or restarted by this code change. Deploy the branch consistently to the host and Spark worker before starting a new spatial trial. Existing runs continue with their recorded source version.

## Verification

Focused spatial, VLA action and recovery tests: 30 passed. The installed robosuite camera utility source was checked for the intrinsic matrix and optical-axis correction conventions. Ruff and Git whitespace checks passed on the changed files. One official RoboCasa playback test was skipped because RoboCasa is not installed in the Lab unit-test environment. One pre-existing depth-schema test was deselected after reproducing its missing-actions failure on the unmodified parent branch. It was not fixed or counted as passing.

The model-to-query-to-model loop was exercised with a scripted caller and mock simulator, including question/answer persistence, token totals and response latency. Native Spark rendering and a real Astra model request have not yet been exercised for this new condition.
