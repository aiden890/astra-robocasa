# Astra: precision actions and destination transit

This branch extends pixel-to-robot distance queries. New runs default to
--motion-control dual; --motion-control legacy retains the original fixed
16-row format. Perception conditions remain independent.

| Mode | Model output | Native execution |
| --- | --- | --- |
| precision | 1 to 4 dataset-order 12-D Cartesian delta rows | Translation norm <=0.20, rotation norm <=0.15, base/torso components <=0.10 |
| transit | Absolute fingertip destination in world XYZ metres | Native OSC_POSE servo until arrival, stall or step limit |

A transit response uses motion_mode=transit, actions=null and
transit_target={position_world_m:[x,y,z],observation_id:current_id}.
A precision response uses motion_mode=precision, actions=[...], transit_target=null.
A depth query uses actions=null, motion_mode=null, transit_target=null.
When queries are enabled, a motion response uses queries=null. scene, progress,
plan and memory fields remain unchanged. Prior demonstration chunks describe
legacy control; the current mode contract takes precedence.

Transit assumes a clear route to a free-space approach point. The executor holds
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

Nine focused control test cases remain: precision worker bounds, host resume without
another model call, destination arrival/lost replies, stale observations, three native
termination cases, interrupted ACK replay and stall-budget preservation. Redundant
schema, configuration, input-shape and overlapping recovery checks were removed.

The physical controller is mocked in this suite. Native Spark speed/reachability
calibration and real model evaluation of destination transit have not been run.
