"""Prepare official object packs separately from assets used by live rollouts."""

import concurrent.futures
import hashlib
import json
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path("/home/csi-agent-dgx_spark2/workspace/astra-assets-complete-20261006")
OLD = Path("/home/csi-agent-dgx_spark2/workspace/rlinf-mibot-first-attempt-440a72b/assets")
PACKS = {
    "aigen_objs": "nwi1vrn5pgbo95kushkasa3nx1i012ff",
    "objaverse": "03eionyo8fk3a9dsksq9jb8du5lqfw8h",
}


def save(path, data):
    """Persist an atomic non-secret progress receipt."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def prepare(name, key):
    """Download and unpack one official pack without touching active assets."""
    report = ROOT / (name + ".json")
    if report.exists() and json.loads(report.read_text()).get("complete"):
        return json.loads(report.read_text())
    url = "https://utexas.box.com/shared/static/" + key + ".zip"
    archive = ROOT / (name + ".zip")
    partial = archive.with_suffix(".partial")
    start = time.time()
    size = 0
    sha = hashlib.sha256()
    if not archive.exists():
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
            total = response.headers.get("Content-Length")
            last = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                sha.update(chunk)
                size += len(chunk)
                if time.time() - last > 5:
                    save(
                        report,
                        {
                            "complete": False,
                            "stage": "download",
                            "source": url,
                            "bytes": size,
                            "total_bytes": total,
                            "elapsed_seconds": time.time() - start,
                        },
                    )
                    last = time.time()
        partial.replace(archive)
    else:
        with archive.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                sha.update(chunk)
                size += len(chunk)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (ROOT / "assets/objects" / member.filename).resolve()
            assert target.is_relative_to((ROOT / "assets/objects").resolve()), member.filename
        z.extractall(ROOT / "assets/objects")
    count = sum(1 for _ in (ROOT / "assets/objects" / name).rglob("model.xml"))
    assert count > 0, (name, "no models")
    result = {
        "complete": True,
        "source": url,
        "bytes": size,
        "sha256": sha.hexdigest(),
        "model_xml_count": count,
        "elapsed_seconds": time.time() - start,
    }
    save(report, result)
    return result


def main():
    """Prepare isolated assets with two bounded download workers."""
    ROOT.mkdir(exist_ok=True)
    (ROOT / "assets/objects").mkdir(parents=True, exist_ok=True)
    for p in OLD.iterdir():
        if p.name != "objects" and not (ROOT / "assets" / p.name).exists():
            (ROOT / "assets" / p.name).symlink_to(p)
    for name in ["lightwheel", "README.md"]:
        p = OLD / "objects" / name
        if p.exists() and not (ROOT / "assets/objects" / name).exists():
            (ROOT / "assets/objects" / name).symlink_to(p)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda kv: prepare(*kv), PACKS.items()))
    save(
        ROOT / "ready.json",
        {
            "download_complete": True,
            "packs": results,
            "assets": str(ROOT / "assets"),
            "time": time.time(),
        },
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        ROOT.mkdir(exist_ok=True)
        save(ROOT / "error.json", {"error": str(error), "time": time.time()})
        raise
