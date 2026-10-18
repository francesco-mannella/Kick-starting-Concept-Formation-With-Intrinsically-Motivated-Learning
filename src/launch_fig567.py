#!/usr/bin/env python
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import launch_run as lr

SRC = Path(__file__).resolve().parent


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Full Figs 5-7 sweep: parasite-mode runs, former-parasite runs started from their"
        " weights at a chosen epoch, disk janitor and progress file"
    )
    parser.add_argument("--root", default="experiments/fig567")
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--pretrain-epoch", type=int, default=1000, help="Parasite checkpoint the former parasite loads")
    parser.add_argument("--epochs-to-test", type=int, default=50)
    parser.add_argument("-o", "--opt", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--min-free-gb", type=float, default=8.0)
    parser.add_argument("--force", action="store_true", help="Delete incomplete run folders")
    return parser.parse_args()


def parse_seeds(text):
    seeds = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        seeds += range(int(lo), int(hi or lo) + 1)
    return seeds


def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def storage(root, mode, seed):
    name = lr.run_name(mode, seed)
    return Path(root).resolve() / name / "simulations" / name / "storage"


def checkpoints(root, mode, seed):
    path = storage(root, mode, seed)
    return sorted(p for p in path.iterdir() if p.is_dir()) if path.exists() else []


def janitor(root, runs):
    freed = 0
    for mode, seed in runs:
        for checkpoint in checkpoints(root, mode, seed)[:-2]:
            for name in ("data.npy", "data_par.npy"):
                path = checkpoint / name
                if path.exists():
                    freed += path.stat().st_size
                    path.unlink()
    return freed


def write_progress(root, state, running, epochs):
    lines = []
    for (mode, seed), status in sorted(state.items()):
        done = checkpoints(root, mode, seed)
        last = int(done[-1].name) + 1 if done else 0
        lines.append(f"{lr.run_name(mode, seed)} {status} {last}/{epochs}")
    free = shutil.disk_usage(root).free / 1e9
    (Path(root) / "progress.txt").write_text(f"{time.strftime('%H:%M:%S')} free {free:.0f} GB\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    args = parse_arguments()
    seeds = parse_seeds(args.seeds)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    pretrained = f"{args.pretrain_epoch:06d}"
    pending = [("parasite", s) for s in seeds] + [("former", s) for s in seeds]
    running, plans, state, started = {}, {}, {}, {}
    last_janitor = last_progress = 0.0

    def ready(mode, seed):
        if mode == "parasite":
            return True
        return (storage(root, "parasite", seed) / pretrained / "weights-parasite.npy").is_file() and (
            state.get(("parasite", seed)) not in ("failed",)
        )

    log(f"{len(pending)} runs, root {root}, {args.epochs} epochs, former from parasite epoch {args.pretrain_epoch}")
    try:
        while pending or running:
            for item in list(pending):
                mode, seed = item
                if state.get(("parasite", seed)) == "failed" and mode == "former":
                    pending.remove(item)
                    state[item] = "blocked"
                    log(f"{lr.run_name(mode, seed)}: blocked (parasite failed)")
                    continue
                if not ready(mode, seed):
                    continue
                if lr.is_done(root, mode, seed, args.epochs):
                    pending.remove(item)
                    state[item] = "done"
                    log(f"{lr.run_name(mode, seed)}: already done")
                    continue
                if shutil.disk_usage(root).free / 1e9 < args.min_free_gb:
                    log(f"low disk, waiting before starting {lr.run_name(mode, seed)}")
                    janitor(root, list(state) + [item])
                    continue
                weights = storage(root, "parasite", seed) / pretrained / "weights-parasite.npy" if mode == "former" else None
                plan = lr.plan_run(root, mode, seed, args.epochs, args.epochs_to_test, args.opt, weights=weights)
                running[item] = lr.start(plan, args.force)
                plans[item], started[item], state[item] = plan, time.time(), "running"
                pending.remove(item)
                log(f"{plan['name']}: started")
            for item, process in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                mode, seed = item
                state[item] = lr.finish(plans[item], code, lr.is_done(root, mode, seed, args.epochs))
                del running[item]
                log(f"{plans[item]['name']}: {state[item]} after {(time.time() - started[item]) / 60:.0f} min")
            now = time.time()
            if now - last_janitor > 300:
                freed = janitor(root, list(state))
                last_janitor = now
                if freed:
                    log(f"janitor freed {freed / 1e9:.1f} GB")
            if now - last_progress > 60:
                write_progress(root, state, running, args.epochs)
                last_progress = now
            time.sleep(10)
    except KeyboardInterrupt:
        log("interrupted, stopping runs")
        for process in running.values():
            process.terminate()
        sys.exit(130)

    for mode, seed in state:
        for checkpoint in checkpoints(root, mode, seed)[:-1]:
            for name in ("data.npy", "data_par.npy"):
                (checkpoint / name).unlink(missing_ok=True)
    write_progress(root, state, running, args.epochs)
    failed = [lr.run_name(m, s) for (m, s), st in state.items() if st != "done"]
    log(f"finished: {sum(st == 'done' for st in state.values())} done, {len(failed)} not done: {failed}")
    if not failed:
        subprocess.run([sys.executable, str(SRC / "plot_fig567.py"), "--db", str(root / "runs.db"), "--out", str(root / "figures")])
    sys.exit(1 if failed else 0)
