# Astra: two motion modes

This branch extends `codex/vla-pixel-robot-distance`. New runs default to
`--motion-control dual`; `--motion-control legacy` keeps the fixed 16-row format.
Perception conditions, including `--condition spatial`, remain independent.

| Mode | Native rows per decision | Translation norm | Rotation norm | Base/torso component |
| --- | --- | --- | --- | --- |
| precision | 1 to 4 | 0.20 | 0.15 | 0.10 |
| transit | 1 to 16 | 1.0 | 1.0 | 1.0 |

The values are normalized native controller inputs, not cm or rad/s. Frequency
remains 20 Hz. These initial bounds need native calibration; they are not proven
collision-safe speeds. Transit permits coarse movement in clear space. Precision
limits individual motions and returns observations sooner near contact or a goal.
It does not use an automatic target servo, path planner, force sensor or obstacle
checker. The model still outputs the 12-D Cartesian delta rows and chooses the
mode, direction and length. Input image/depth/relative-distance queries are unchanged.

The native worker enforces bounds even if the policy exceeds them. It preserves
the gripper and arm/base selector: the new motion_mode field is independent of
index 4, which still switches arm/base control. A gripper may need 10 to 16 native
steps, so a precision grasp may span several model turns.

An action response adds `"motion_mode": "precision"` or `"transit"` and supplies
1..4 or 1..16 rows respectively. A query response uses actions=null and
motion_mode=null. The model sees the enforced limits in its system prompt.
Config stores the control setting and profiles. Progress stores the selected
mode and bounded actions; original model output stays in the call response/trace.
Worker feedback stores the actual executed actions and mode. The checkpoint request
identity includes the mode, so a lost acknowledgement cannot silently switch
profiles. Resume adopts the original config and rejects configuration/source changes.
Existing experiments must continue from their original source commit; do not
restart or migrate them to this branch.

## Validation

Focused action, spatial-query and recovery tests: 40 passed, 1 skipped (official
RoboCasa import unavailable), 1 deselected (pre-existing depth schema fixture
failure reproduced on the parent branch). Scoped Ruff check/format and git diff
checks pass. Mock host interruption reuses a precision reply and transitions
to transit without an extra call; original requested rows and usage remain recorded.
Native speed calibration and real model evaluation have not been run.
