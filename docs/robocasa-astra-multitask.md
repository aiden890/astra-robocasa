# Native multi-task Astra demos

The task manifest records the 31 requested RoboCasa tasks and full native step budgets.
Each task has one PandaOmron run and one GR1FloatingBody run. The queue runs at
most four simulators in total on Spark2, including adopted workers started by a
previous supervisor. Recordings are 20 fps and remain on Lab-desktop.

GR1 initial orientation uses the native task starting fixture. The base yaw faces
the fixture, with bounded backward clearance when the rotated body collides.
Initialization records the fixture, initial and final base position, yaw change,
heading error, and collision state. A heading error greater than five degrees or
remaining robot collision rejects the initialization. Objects and native success
conditions are retained.

Some existing object packs were incomplete. Official object archives are prepared
in a separate Spark2 directory, preserving active mounts. The asset activation
worker waits for both archives, then checks native GR1 boot, reset, rendering and
two no-op steps for PanTransfer, TurnOnMicrowave and CoffeeSetupMug. These checks
make no model calls and do not measure task success. Only passing checks enable
new containers for future queued runs. Setup errors keep the gate closed and are
saved in asset-activation-error.json.

Runtime state: .runtime/multitask-20261006-v1/status.json and asset-setup.json.
Public progress: status-page/media/task-queue.json and the task video topics.
Initialization failures keep their original output and receive separate retry IDs.
Task success is the native success_at_end metric from a completed official evaluation.
A model capacity error is an execution error, not a measured task failure.

Official asset registry:
https://raw.githubusercontent.com/robocasa/robocasa/main/robocasa/models/assets/box_links/box_links_assets.json
