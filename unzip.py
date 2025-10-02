# mmsafety_unzip.py  — unzip the image pack to MM-SafetyBench/data/imgs
import os, zipfile, shutil
from pathlib import Path
from collections import defaultdict

# --- configure these two paths ---
ZIP_PATH  = Path("MM-SafetyBench_imgs.zip")     # path to your downloaded zip
REPO_ROOT = Path("MM-SafetyBench")              # path to the cloned repo root
# ---------------------------------

DEST_DIR = REPO_ROOT / "data" / "imgs"
DEST_DIR.mkdir(parents=True, exist_ok=True)

if not ZIP_PATH.exists():
    raise FileNotFoundError(f"Zip not found: {ZIP_PATH.resolve()}")
if not REPO_ROOT.exists():
    raise FileNotFoundError(f"Repo root not found: {REPO_ROOT.resolve()}")

def _is_dir_member(name: str) -> bool:
    return name.endswith("/") or name.endswith("\\")

def _rel_after_imgs(name: str) -> str | None:
    # return path *after* the first 'imgs/' segment, or None
    parts = Path(name).as_posix().split("/")
    for i, p in enumerate(parts):
        if p == "imgs":
            return "/".join(parts[i+1:])
    return None

extracted = 0
skipped_dirs = 0

with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    # First pass: detect whether entries are under 'imgs/' or directly '<scenario>/<variant>/...'
    members = [m for m in zf.infolist() if not _is_dir_member(m.filename)]
    has_imgs_prefix = any(_rel_after_imgs(m.filename) for m in members)

    for info in members:
        src = info.filename
        if has_imgs_prefix:
            rel = _rel_after_imgs(src)
            if rel is None or rel == "":
                continue
            out_path = DEST_DIR / Path(rel)
        else:
            # assume entries are already '<scenario>/<variant>/...'
            out_path = DEST_DIR / Path(src)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info, "r") as fsrc, open(out_path, "wb") as fdst:
            shutil.copyfileobj(fsrc, fdst)
        extracted += 1

# Summary
by_scenario = defaultdict(lambda: defaultdict(int))
for p in DEST_DIR.rglob("*.jpg"):
    parts = p.relative_to(DEST_DIR).as_posix().split("/")
    if len(parts) >= 3:
        scen, variant = parts[0], parts[1]
        by_scenario[scen][variant] += 1

print(f"Extracted files: {extracted}")
if not by_scenario:
    print(f"WARNING: no JPGs found under {DEST_DIR}. Check the zip structure and paths.")
else:
    print(f"Destination: {DEST_DIR.resolve()}")
    print("Counts per scenario/variant:")
    for scen in sorted(by_scenario):
        row = ", ".join(f"{v}:{by_scenario[scen][v]}" for v in sorted(by_scenario[scen]))
        print(f"  {scen} -> {row}")
