### Added
- Independent Spark2 simulator containers and a staged two-to-four-run coordinator that adopts an existing rollout and records aggregate wall-clock throughput.
- Per-run container selection and throughput accounting tests.
### Fixed
- GR1 environment creation no longer supplies ignore_done twice to RoboCasa's helper.
