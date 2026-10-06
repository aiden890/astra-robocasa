# Rollout status page

Main and video tabs reuse the user's RoboCasa tracking page template. The main
tab stays empty. The video board supports robot/task/outcome filters, search,
ten-row pagination, and a separate video player. Refreshing the catalog does
not reload the player.

`publish_videos.py` exports every synchronized post-action camera frame from
Inspect Robots' NPY sidecars at 20fps. No interpolation or snapshot repetition
is used. Native control is configured and checked at 20Hz. Video duration is
simulation time, excluding model latency. Original sidecars, actions, logs, and
previous snapshot previews remain preserved.

`run_full_queue.py` runs PandaOmron then GR1FloatingBody with a maximum of 1800
steps each and native success termination. Existing output directories cannot
be reused. The catalog refreshes every 15 seconds; completed/error trials have
validated synchronized recordings exported automatically.

Install imageio-ffmpeg into the private Lab runtime to encode recordings. Media
is ignored by Git. Serve only this directory, never the repository root.

## Topic library

The board now lists topics, not individual runs. `topics.json` stores titles,
descriptions, categories, model/environment, date, tags, and matching conditions
or explicit run IDs. `topic.html?topic=<id>` groups multiple recordings and
shows their per-run status, horizon, frame rate, cameras, and MP4 links.
Search spans descriptions and tags as well as run IDs and robot names.
Uncategorized runs remain visible. Auto-refresh preserves mounted video
elements and playback. The main tab remains empty.
