# Camera depth for Astra

## Observation and model input

Each available camera is rendered once with `depth=True`, producing RGB and
normalized depth at the same simulation step. The worker converts the depth
using robosuite `get_real_depth_map`, then flips both arrays vertically into
top-left pixel order. H×W and H×W×1 renderer outputs are supported. Invalid
normalized values fail explicitly. Metric values are finite positive float32
camera optical-axis distances in meters, not Euclidean distance from the camera.

The camera metadata includes the simulation step and time, near/far planes,
intrinsic matrix, and camera-to-world transform. The worker transports metric
arrays as lossless compressed NPY through the existing JSON-line connection.
The Lab bridge stores them under `worker/depth/observation-NNNNNN/`, alongside
metadata. No depth frames or videos are recorded to Spark2 disk.

The policy receives each original RGB image plus a named depth preview image
through the actual Codex image arguments. Previews map 0–3 meters from white
to black; distances beyond 3 meters saturate only in the preview. The prompt
also includes an 8×8 grid of metric distances with explicit pixel coordinates.
Raw arrays and calibration are copied into each model-call record. The model
receives PNG previews and text samples; a raw NPY is not a native model input.
Panda normally supplies three RGB and three depth images; GR1 normally supplies
four RGB and four depth images. Actual available camera names determine the count.

Uncompressed metric storage is approximately 1 MiB per four-camera observation
(256×256×4 bytes per camera), before metadata and model-call copies. Depth
observations occur at policy observation boundaries, not every 20 Hz video frame.

## Invocation and remote package

The runner enables depth by default on this branch. `--no-depth` retains RGB-only
rendering. Direct worker calls require `--depth`. The queue stages the complete
branch worker package into a new temporary directory in an idle leased container
and passes its path with `--worker-pythonpath`. This avoids mixing a new depth
worker with stale modules from the existing mounted deployment. Asset activation
also stages the complete package when preparing a new container. Completed asset
activations and active simulator processes are not restarted by this change.

## RPent comparison

Audited RPent commit `068cd64f4f175179a13c67d429a4d00ca56fb196`:

- [env_server.py](https://github.com/RLinf/RPent/blob/068cd64f4f175179a13c67d429a4d00ca56fb196/robots/robocasa/env_server.py):
  `render_camera` requests RGB plus depth and converts to meters; camera metadata
  and transforms are exposed.
- [env_client.py](https://github.com/RLinf/RPent/blob/068cd64f4f175179a13c67d429a4d00ca56fb196/robots/robocasa/env_client.py):
  RGB and metric depth are flipped together, then depth is back-projected to
  a pixel-aligned world-coordinate map.
- [tools.py](https://github.com/RLinf/RPent/blob/068cd64f4f175179a13c67d429a4d00ca56fb196/robots/robocasa/tools.py):
  observation recording preserves depth and world maps; back-projection tools
  query those maps.

Our capture, metric conversion and orientation follow the same robosuite
semantics. RPent repairs non-finite renderer values to the far plane; this branch
rejects corrupted values instead. This branch provides metric observations and
calibration, but does not implement RPent's pixel-to-world query tools. No RPent
source was copied. Calibration availability alone is not tool equivalence.

## Verification and limits

31 plugin and operational tests passed, including lossless transport, invalid
depth rejection, same-render RGB/depth orientation, bridge-to-policy image and
metric sample delivery, and full remote-package staging. Scoped Ruff checks passed.

An isolated Lab CPU OSMesa test used real robosuite 1.5.2 and MuJoCo 3.3.1 with
a known geometric scene and all five camera names used across the two robots.
A surface at 1.8 meters measured 1.8000145 m; a floor at 2 meters measured
2.0000019 m. Back-projection using the saved calibration recovered world z
0.1999855 m for a surface at 0.2 m. RGB was identical to RGB-only rendering and
saved metric arrays were exactly equal to transported arrays.
The [raw report](../reports/depth/native-renderer-20261006.json) is preserved.

The initial isolated EGL attempt could not initialize the Lab graphics device;
the successful test used a separately extracted OSMesa library without modifying
the host installation. No model calls, GPU workloads, or native RoboCasa rollouts
were launched by the validation. Full RoboCasa RGB-depth rollout and subscription
model behavior remain unverified. Existing production episodes remain RGB-only.
