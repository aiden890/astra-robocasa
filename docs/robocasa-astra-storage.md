# Recording storage

Spark2 runs the simulator and sends observations over the private SSH connection. Camera images are encoded in memory for each response. The worker does not save video files or frame arrays on Spark2.

Lab-desktop owns the run directories, model call receipts, Inspect evaluation logs, saved camera frames, encoded 20fps MP4 files, and the video board. Current runs live under /home/aiden/Desktop/lab/robot/astra-robocasa/runs; published videos live under its status-page/media directory.

Historical Spark2 recordings are archived on Lab-desktop under /home/aiden/Desktop/lab/robot/astra-robocasa-media-archive/spark2-20261006. The files directory preserves their original relative workspace paths. manifest.json records every original size, timestamp, and SHA256. verified.json records the successful Lab verification. Spark2 copies are removed only after every archived file is verified, with a final source checksum and timestamp check before deletion. Runtime code, scene fixtures, meshes, textures, model checkpoints, and active processes are retained.
