# Astra: precision actions and destination transit

This branch extends pixel-to-robot distance queries. New runs default to
--motion-control dual; --motion-control legacy retains the original fixed
16-row format. Perception conditions remain independent.

| Mode | Model output | Native execution |
| --- | --- | --- |
| precision | 1 to 4 dataset-order 12-D Cartesian delta rows | Translation norm <=0.20, rotation norm <=0.15, base/torso components <=0.10 |
| transit | Absolute fingertip destination in world XYZ metres | Native OSC_POSE servo until arrival, stall or step limit |

A transit response uses motion_mode=transit, actions=null and
transit_target={position_world_m:[x,y,z],observation_id:current_id,
purpose:pre_precision|transport,grasp_confirmed:false|true}.
A precision response uses motion_mode=precision, actions=[...], transit_target=null.
A depth query uses actions=null, motion_mode=null, transit_target=null.
When queries are enabled, a motion response uses queries=null. scene, progress,
plan and memory fields remain unchanged. Prior demonstration chunks describe
legacy control; the current mode contract takes precedence.

## Mode selection policy

Precision handles actual contact and fine manipulation: alignment, grasping/regrasp,
turning or pushing, insertion, release and small corrections. Transit handles other
destination travel: approach to the position immediately before a precision operation
(pre_precision), or transport of an object already securely grasped (transport).
Holding a secured object during travel does not itself require precision.

Intermediate obstacles and external disturbances are absent by experimental
assumption during transit. This implementation adds no obstacle planner. A transit
ends at a pre-manipulation destination; the model observes the outcome and uses
precision for the next contact, fine adjustment or release.

For transport the model must confirm the actual grasp from current observations and
feedback, explaining the evidence in progress. Closing the gripper alone is not grasp
confirmation. The output contract requires grasp_confirmed=true for transport; the
worker also rejects transport with an open gripper command before any native step.
These checks enforce the declared policy, not an independent physical attachment
detector. pre_precision may use either gripper state and does not require a grasp.
If attachment is uncertain, use precision to establish or correct the grasp.

Transit assumes a clear route to a pre-manipulation destination. The executor holds
the initial fingertip orientation, gripper command and stationary base, using
the existing physical OSC controller. It does not teleport joints, add an IK
solver or plan around obstacles. The model does not supply low-level transit
rows or make additional calls during a single destination move. Orientation and
gripper changes remain precision actions.

Arrival requires <=1 cm position error and <=5 degrees orientation error.
Stop after 160 native steps, 20 steps without 2 mm progress, task success or the
episode step budget. Frequency stays 20 Hz. Native controller inputs are capped
at 0.5 per component by the existing arm servo. No arrival guarantee is made for
unreachable or blocked destinations. Feedback reports target_reached, remaining
position/orientation error, stop_reason, actual native actions and steps.

Each target binds its source observation ID. Observations expose current
fingertip_world_m, independent of depth. Spatial queries return surface world
coordinates; choose a free-space offset rather than the measured surface itself.
No hidden object coordinates are added.

## Recovery and provenance

The worker checkpoints the immutable absolute destination, initial orientation,
gripper command and every acknowledged native action. Restoring the frozen scene
replays acknowledged actions and validates state digests before continuing the
same target. The remaining servo step cap and stall history survive interruption.
A completed request returns its saved response without moving again. Reusing a
request ID with a different destination is rejected.

Source/config identities distinguish this contract from previous dual chunk runs.
Existing experiments must keep their recorded source commit. Do not migrate their
checkpoints. Raw model output, usage and latency remain in original receipts and
trace. Model response and episode wall time have no fixed timeout.

## Validation

Seventeen essential related cases remain across the five VLA test files.
Nine control cases cover: precision worker bounds, host resume without
another model call, destination arrival/lost replies, stale observations, three native
termination cases, interrupted ACK replay and stall-budget preservation. Redundant
schema, configuration, input-shape and overlapping recovery checks were removed.
The other eight cover native action mapping, checkpoint digest/replay integrity,
detached model calls, retry accounting and budget, spatial projection, frozen
observation guards and query-answer delivery to the model. There are no excluded
or skipped cases in this focused set.

The physical controller is mocked in this suite. Native Spark speed/reachability
calibration and real model evaluation of destination transit have not been run.
