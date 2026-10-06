# Research video board

The video list links each research topic to its own detail page. Topics can contain several recordings with robot, task, model, date, tags, and a description.

The layout follows the project-oriented organization of Oier Mees's personal research page (https://www.oiermees.com/): a searchable research title leads to a dedicated page with related media. No reference text or media is copied.

Deletion moves a topic or recording to a shared, persistent trash list. Restore is available from the video board. Original recording files and running experiments remain intact. Visibility state is stored outside the public directory in `.runtime/status-page/board-state.json`.

Serve the page using `astra_ops.media.board`, with the public `status-page` directory and a private state path. The API accepts same-origin requests and validates item identifiers. Do not serve the repository root.
