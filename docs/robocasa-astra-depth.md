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
through the actual Codex image arguments. Previews map the current camera's
2nd–98th depth percentiles from white to black. The prompt includes the actual
display range in meters and a 16×16 grid of metric distances with explicit pixel
coordinates. Smaller images use fewer unique sample coordinates.
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

## Camera-only input contract

The initial input uses each camera's RGB PNG followed by a named 8-bit RGB
greyscale depth PNG at 256×256. The depth preview uses relative contrast for
each camera and observation: the 2nd percentile is white and the 98th is black.
Values outside that range saturate in the image, and a constant range is shown
as mid-grey. Each prompt records both endpoints in meters and warns that equal
shades across cameras or observations do not imply equal distances.
Quantization applies only to the preview; float32 NPY remains the metric source
of truth. The 16×16 text grid supplements the image with meters.
No world-coordinate query is included. Depth is the first visible surface's
optical-axis Z distance, not an object center or a straight-line range.

### Visualization references

Linear clipped normalization follows the contract of
[Matplotlib Normalize](https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.Normalize.html).
The sequential reversed grayscale palette (`gray_r`) keeps nearer surfaces
brighter, following the guidance for ordered measurements in
[Matplotlib's colormap guide](https://matplotlib.org/stable/users/explain/colors/colormaps.html).
The percentile range is selected with
[NumPy percentile](https://numpy.org/doc/stable/reference/generated/numpy.percentile.html),
and the RGB preview is rendered using
[Pillow ImageOps.colorize](https://pillow.readthedocs.io/en/stable/reference/ImageOps.html#PIL.ImageOps.colorize).
Matplotlib is a design reference, not a runtime dependency. Preview generation
uses the existing NumPy and Pillow dependencies and never changes raw depth.
The choice of 2nd–98th percentiles is this project's contrast setting, not a
robotics benchmark standard. Previously published preview videos retain their
original display scale; this contract applies to newly generated previews.

For Panda the camera order is agentview_left, agentview_right, eye_in_hand.
For GR1 it is agentview_left, agentview_right, eye_in_right_hand,
eye_in_left_hand. Preview names and image attachment order are spelled out in
the prompt so each depth image can be matched with its RGB camera.

An [eight-second video preview](http://100.86.183.64:8906/topic.html?topic=camera-depth-preview)
is published in the video board. It contains 160 real CPU-rendered frames at
20 fps, with RGB on the upper row and depth on the lower row. The moving
geometric scene illustrates the sensor format; it is not a RoboCasa rollout
or an Astra inference result. The isolated command is
`python -m astra_ops.media.depth_preview --output NEW_DIRECTORY` with a
compatible CPU MuJoCo renderer. It never overwrites an existing output directory.

Supplemental diagnostic records are kept in `status-page/media/supplemental-catalog.json`,
separate from the queue-generated catalog. The video board tolerates an absent
supplemental file and honors the existing delete controls for these entries.

### Native RoboCasa preview

`astra_ops.media.robocasa_depth_preview` uses a reserved idle container on
`--host` (default Spark2). It resets the actual PrepareCoffee environment with
PandaOmron, records three synchronized cameras for 60 frames, and sends zero
controller actions for 59 steps. It does not call Astra or assign a task score.
All RGB and float32 metric observations are stored on Lab-desktop. The initial
CPU run needs both `MUJOCO_GL=osmesa` and `PYOPENGL_PLATFORM=osmesa`; it does not
require a GPU. Host choice does not alter the production rollout queue.

The actual [RoboCasa camera preview](http://100.86.183.64:8906/topic.html?topic=robocasa-depth-preview)
was rendered on Spark2 after a native job released its slot. New dispatches were
briefly held while all existing episodes continued, keeping at most four active
Spark2 simulators. Dispatch resumed automatically afterward. The clip decodes to
60 frames at 20 fps (3 seconds), RGB above depth, three Panda cameras. The
[publication report](../reports/depth/robocasa-preview-20261006.json) records the
video checksum and unscored sensor-only result. No model calls were made.

The user requested Spark2 only. The earlier AMP CPU preparation was cancelled,
its owned temporary container and incomplete asset copies were removed, and its
failed initialization logs were retained on Lab. No existing AMP workloads were
changed. Future previews for this project use Spark2.

### Moving robot preview

`--motion` renders 160 frames at 20 fps while applying a bounded scripted arm
lift, lateral movement and return in the native PrepareCoffee environment. The
clip is [published separately](http://100.86.183.64:8906/topic.html?topic=robocasa-depth-motion).
The three synchronized depth streams changed and the calibrated wrist-camera
position moved up to 0.025337 m. The initial EEF-displacement calculation mixed
relative and absolute state fields; that metric is explicitly invalidated in
the publication report. Motion was verified independently from the saved
per-frame camera transforms. Future recordings select one stable named EEF
field. The original report is preserved in runtime.

To keep four active Spark2 simulators, one owned existing rollout was temporarily
paused in place along with dispatch. Its simulator state and process IDs were
preserved and resumed after the preview. The preview's separate temporary
container was removed. No Astra calls, task scoring or policy changes were made.
