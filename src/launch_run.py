#!/usr/bin/env python
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
MODES = ("parasite", "former", "agent")


def run_name(mode, seed, tag=None):
    return f"{mode}_{tag}_{seed:03d}" if tag else f"{mode}_{seed:03d}"


def run_dir(root, mode, seed, tag=None):
    return Path(root).resolve() / run_name(mode, seed, tag)


def final_files(root, mode, seed, epochs, tag=None):
    storage = run_dir(root, mode, seed, tag) / "simulations" / run_name(mode, seed, tag) / "storage"
    last = storage / f"{epochs - 1:06d}"
    files = [last / "weights.npy"]
    if mode == "parasite":
        files.append(last / "weights-parasite.npy")
    return files


def is_done(root, mode, seed, epochs, tag=None):
    return all(f.is_file() for f in final_files(root, mode, seed, epochs, tag))


def plan_run(root, mode, seed, epochs, epochs_to_test, options=(), weights=None,
             parasite_epochs=None, check_weights=True, tag=None, db=None):
    name = run_name(mode, seed, tag)
    command = [
        sys.executable, str(SRC / "SMMain.py"),
        "-s", str(seed), "-n", name, "--group", tag or mode,
        "--local", "--db", str(Path(db).resolve() if db else Path(root).resolve() / "runs.db"),
        "-o", f"epochs={epochs}", "-o", f"epochs_to_test={epochs_to_test}",
    ]
    if mode == "parasite":
        command.append("--parasite")
    if mode == "former":
        if weights is None:
            weights = final_files(root, "parasite", seed, parasite_epochs or epochs)[1]
        if check_weights and not Path(weights).is_file():
            raise FileNotFoundError(f"parasite weights not found: {weights}")
        command += ["-l", str(weights)]
    for option in options:
        command += ["-o", option]
    return {"name": name, "command": command, "cwd": run_dir(root, mode, seed, tag)}


def start(plan, force=False):
    cwd = plan["cwd"]
    if cwd.exists() and any(cwd.iterdir()):
        if not force:
            raise FileExistsError(f"{cwd} is not empty (incomplete run); use --force to delete it")
        shutil.rmtree(cwd)
    cwd.mkdir(parents=True)
    (cwd / "status").write_text("running\n")
    env = dict(os.environ, OMP_NUM_THREADS="1")
    log = open(cwd / "sim.log", "w")
    return subprocess.Popen(plan["command"], cwd=cwd, env=env, stdout=log,
                            stderr=subprocess.STDOUT)


def finish(plan, returncode, ok):
    status = "done" if returncode == 0 and ok else "failed"
    (plan["cwd"] / "status").write_text(status + "\n")
    return status


def parse_arguments():
    parser = argparse.ArgumentParser(description="Launch a single simulation run")
    parser.add_argument("--mode", choices=MODES, required=True,
                        help="parasite: active agent + parasite; former: active agent"
                        " starting from the parasite's maps; agent: plain active agent")
    parser.add_argument("-s", "--seed", type=int, required=True)
    parser.add_argument("--root", default="experiments/figs567",
                        help="Folder holding run folders and runs.db")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--epochs-to-test", type=int, default=10)
    parser.add_argument("--parasite-epochs", type=int, default=None,
                        help="Epochs of the parasite run whose weights are loaded (former)")
    parser.add_argument("--weights", default=None, help="Explicit weights file (former)")
    parser.add_argument("-o", "--opt", action="append", default=[], metavar="KEY=VALUE",
                        help="Extra SMMain option")
    parser.add_argument("--db", default=None, help="Database file (default: <root>/runs.db)")
    parser.add_argument("--tag", default=None, help="Condition label added to the run name and used as group")
    parser.add_argument("--force", action="store_true", help="Delete an incomplete run folder")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if is_done(args.root, args.mode, args.seed, args.epochs, args.tag):
        print(f"{run_name(args.mode, args.seed, args.tag)}: already done")
        sys.exit(0)
    plan = plan_run(args.root, args.mode, args.seed, args.epochs, args.epochs_to_test,
                    args.opt, args.weights, args.parasite_epochs,
                    check_weights=not args.dry_run, tag=args.tag, db=args.db)
    print(f"cd {plan['cwd']} && {' '.join(plan['command'])}")
    if args.dry_run:
        sys.exit(0)
    process = start(plan, args.force)
    try:
        code = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        code = process.wait()
    status = finish(plan, code, is_done(args.root, args.mode, args.seed, args.epochs, args.tag))
    print(f"{plan['name']}: {status} (log: {plan['cwd'] / 'sim.log'})")
    sys.exit(0 if status == "done" else 1)
