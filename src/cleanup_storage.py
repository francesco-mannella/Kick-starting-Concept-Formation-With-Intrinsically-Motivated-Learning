#!/usr/bin/env python
import argparse
import os
import re
import shutil
from pathlib import Path


def dir_size(path):
    total = 0
    for base, _, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(base, name)).st_size
            except OSError:
                pass
    return total


def epoch_dirs(storage):
    return sorted(d for d in storage.iterdir() if d.is_dir() and re.fullmatch(r"\d{6}", d.name))


def finished_storages(root):
    for run in sorted(Path(root).iterdir()):
        status = run / "status"
        if run.is_dir() and status.is_file() and status.read_text().strip() == "done":
            yield from sorted(run.glob("simulations/*/storage"))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Delete per-epoch storage folders of finished runs, keeping the"
        " final epoch and every Nth epoch. Dry run unless --apply is given."
    )
    parser.add_argument("root", help="Experiment folder holding the run folders")
    parser.add_argument("--every", type=int, default=100)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    freed = kept = 0
    for storage in finished_storages(args.root):
        dirs = epoch_dirs(storage)
        if not dirs:
            continue
        final = dirs[-1]
        doomed = [d for d in dirs if d != final and int(d.name) % args.every != 0]
        size = sum(dir_size(d) for d in doomed)
        freed += size
        kept += len(dirs) - len(doomed)
        print(f"{storage.parent.name}: keep {len(dirs) - len(doomed)}/{len(dirs)} epochs,"
              f" {'free' if args.apply else 'would free'} {size / 1e9:.1f} GB")
        if args.apply:
            for d in doomed:
                shutil.rmtree(d)
    print(f"total {'freed' if args.apply else 'would free'}: {freed / 1e9:.1f} GB,"
          f" {kept} epoch folders kept")
