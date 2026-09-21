"""Gather the usdz for an arbitrary scene list into one sceneset the renderer can glob.

The renderer serves only what `--artifact-glob` matched, and a sceneset directory holds
whatever run materialised it -- the hard100 shards left four 25-scene sets, so no single
one covers a selection drawn from the whole suite. Rather than run the renderer once per
sceneset, this hardlinks the needed files into one directory.

Hardlinks, not copies: same filesystem, so this costs no space, and the files are shared
inodes with the existing scenesets (deleting this directory frees nothing and breaks
nothing). The name is prefixed so a cleanup can never touch another member's sceneset.
"""
import argparse
import csv
from pathlib import Path

ARTIFACTS = Path("/mnt/nvme1n1/ad_vla/data/nre-artifacts/scenesets")
SUITES_CSV = Path("/home/cvlab21/project/chan/alpasim/data/scenes/sim_suites.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=Path, required=True, help="one scene_id per line")
    ap.add_argument("--name", required=True, help="sceneset directory name (chan_ prefixed)")
    ap.add_argument("--parent-suite", default="public_2601")
    args = ap.parse_args()

    if not args.name.startswith("chan_"):
        raise SystemExit("--name 은 chan_ 으로 시작해야 합니다 (공용 트리 보호)")

    uuid_of = {}
    with open(SUITES_CSV) as fh:
        for row in csv.DictReader(fh):
            if row.get("test_suite_id") == args.parent_suite:
                uuid_of[row["scene_id"]] = row["uuid"]

    scenes = [ln.strip() for ln in args.scenes.read_text().splitlines() if ln.strip()]
    dest = ARTIFACTS / args.name
    dest.mkdir(parents=True, exist_ok=True)

    found, missing = 0, []
    for s in scenes:
        u = uuid_of.get(s)
        if not u:
            missing.append((s, "suite csv 에 없음"))
            continue
        tgt = dest / f"{u}.usdz"
        if tgt.exists():
            found += 1
            continue
        src = next((p for p in ARTIFACTS.glob(f"*/{u}.usdz")
                    if p.parent != dest), None)
        if src is None:
            missing.append((s, f"usdz 없음 ({u})"))
            continue
        tgt.hardlink_to(src)
        found += 1

    print(f"{dest}\n  씬 {found}/{len(scenes)} 준비")
    for s, why in missing:
        print(f"  누락 {s}: {why}")
    print(f"\nSCENESET={args.name}")
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
