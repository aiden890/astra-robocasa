"""Number the common frozen scenes per task (0-9, in evaluation/seeds.json order) and write the index.

    python3 scripts/robocasa-astra/index_scenes.py /data/astra-robocasa-scenes/common-eval-panda-diverse-scenes-20261006

Writes plugins/.../astra_robodawn/assets/scenes.json (scene folders relative to the collection root,
which runs locate through ASTRA_SCENE_ROOT) and context_test_docs/scenes.md. No simulator is started.
"""

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SEEDS = REPO / "evaluation" / "seeds.json"
INDEX = REPO / "plugins/inspect-robots-robocasa-astra/src/robocasa_astra/astra_robodawn/assets/scenes.json"
DOC = REPO / "context_test_docs" / "scenes.md"


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    seeds = json.loads(SEEDS.read_text())["tasks"]
    index = {"collection": root.name, "source": "https://github.com/aiden890/astra-robocasa/releases/tag/common-scenes-diverse-v2-20261006",
             "numbering": "scene n = n-th seed of evaluation/seeds.json for the task (0-based)", "tasks": {}}
    lines = ["# Common scenes (numbered)", "",
             f"Collection `{root.name}` ({index['source']}). Scene *n* of a task is the *n*-th seed in "
             "`evaluation/seeds.json` (0-based). Runs restore a scene exactly with `--scene n` "
             "(root folder in `ASTRA_SCENE_ROOT`).", "",
             "| task | scene | seed | layout / style | horizon | build check: state / images | instruction |",
             "|---|---|---|---|---|---|---|"]
    for task, task_seeds in seeds.items():
        rows = []
        for n, seed in enumerate(task_seeds):
            folder = root / "scenes" / f"{task}-{seed}"
            manifest = json.loads((folder / "manifest.json").read_text())
            meta = json.loads((folder / "episode-meta.json").read_text())
            verified = json.loads((folder / "verified.json").read_text())
            row = {"scene": n, "scene_id": folder.name, "folder": f"scenes/{folder.name}", "rollout_seed": seed,
                   "simulator_seed": manifest["simulator_seed"], "horizon": manifest["horizon"],
                   "layout_id": meta["layout_id"], "style_id": meta["style_id"], "instruction": meta["lang"],
                   "manifest_sha256": hashlib.sha256((folder / "manifest.json").read_bytes()).hexdigest(),
                   "build_state_exact": verified.get("state_exact"),
                   "build_initial_images_exact": verified.get("initial_images_exact")}
            rows.append(row)
            lines.append(f"| {task} | {n} | {seed} | {row['layout_id']} / {row['style_id']} | {row['horizon']} | "
                         f"{row['build_state_exact']} / {row['build_initial_images_exact']} | {row['instruction']} |")
        index["tasks"][task] = rows
    INDEX.write_text(json.dumps(index, indent=1))
    DOC.write_text("\n".join(lines) + "\n")
    print(f"wrote {INDEX} and {DOC}")


if __name__ == "__main__":
    main()
