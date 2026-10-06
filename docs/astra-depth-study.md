# Paired camera-depth study

The study evaluates four input conditions on the same immutable PandaOmron snapshots:
RGB only, RGB plus aligned depth previews, RGB plus pixel camera-Z queries, and RGB
plus 16×16 camera-Z cell queries. Each condition covers ten scenes in each of
PrepareCoffee, PanTransfer, OpenCabinet, PickPlaceSinkToCounter and StirVegetables:
200 independent rollouts. Native complete-task success, controller, scene horizon
and 20 Hz control remain unchanged.

`robocasa_common.depth_trial` runs one condition in a separate process.
`robocasa_common.depth_study` owns the background queue and waits for the existing
Xiaomi first pass and error recovery to finish. A native depth and actual-model-call
preflight receipt is required. It starts with eight slots and ramps toward 32 only
when measured memory, CPU, latency and provider errors permit; 32 is a candidate
ceiling rather than a validated capacity claim. Active trials are never terminated
by a concurrency reduction. Partial outputs are retained as execution errors.

All calls explicitly select `gpt-6-astra` and `medium` using the existing subscription.
Robot proprioception and RGB are common inputs. Camera depth is private in the
RGB-only condition. Queries use top-left pixel `(column, row)` coordinates, match
an exact observation ID and hold the robot still. Cell distributions and their
center samples are distinct. Distances are optical-axis Z in meters, not world
coordinates or Euclidean range.

Raw structured CLI events, prompts, outputs and attempt receipts are retained
privately. Input, cached-input and output usage are read from completed-turn events;
cached input is already part of input and is not counted again. Missing usage is
unknown. Monotonic CLI end-to-end latency includes startup, network, provider queue
and generation; server GPU time and a separate reasoning-token count are unavailable.
Every retry and distance-query round contributes to the measured call budget.

`robocasa_common.depth_site` publishes the protocol, measured totals, native results,
paired comparisons and verified 768×256/20 fps videos. Media is encoded on Lab directly
without per-step frame files; Spark holds only read-only code and scene assets.

Camera-Z arrays in memory-only mode are retained only for the current observation,
outside persistent per-step extras. A matching simulation-step identity is required
before a query or preview reads them. Exact queried float32 regions and their
checksums are saved separately. This prevents whole-image depth arrays from
accumulating in trial history; model inputs and query values are unchanged.
