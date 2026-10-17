#!/usr/bin/env python
import argparse
import sys
import time

import launch_run as lr


def parse_seeds(text):
    seeds = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds += range(int(a), int(b) + 1)
        else:
            seeds.append(int(part))
    return seeds


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Launch the runs for Figs 5-7: per seed a parasite run, then a"
        " former-parasite run started from its maps"
    )
    parser.add_argument("--root", default="experiments/figs567")
    parser.add_argument("--seeds", default="0-9", help="e.g. 0-9 or 0,3,5")
    parser.add_argument("-p", "--parallel", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--former-epochs", type=int, default=None,
                        help="Epochs of the former-parasite runs (default: --epochs)")
    parser.add_argument("--epochs-to-test", type=int, default=10)
    parser.add_argument("--stages", default="parasite,former")
    parser.add_argument("-o", "--opt", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--force", action="store_true", help="Delete incomplete run folders")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


if __name__ == "__main__":
    args = parse_arguments()
    seeds = parse_seeds(args.seeds)
    stages = args.stages.split(",")
    former_epochs = args.former_epochs or args.epochs
    epochs = {"parasite": args.epochs, "former": former_epochs}

    pending = [(stage, s) for stage in ("parasite", "former") if stage in stages for s in seeds]
    state = {}
    running = {}
    started = {}

    def parasite_ready(seed):
        if lr.is_done(args.root, "parasite", seed, args.epochs):
            return True
        return None if ("parasite", seed) in state or ("parasite", seed) in running else False

    if args.dry_run:
        for stage, seed in pending:
            plan = lr.plan_run(args.root, stage, seed, epochs[stage], args.epochs_to_test,
                               args.opt, parasite_epochs=args.epochs, check_weights=False)
            print(f"cd {plan['cwd']} && {' '.join(plan['command'])}")
        sys.exit(0)

    log(f"{len(pending)} runs, {args.parallel} in parallel, root {args.root}")
    try:
        while pending or running:
            for job in list(pending):
                if len(running) >= args.parallel:
                    break
                stage, seed = job
                if lr.is_done(args.root, stage, seed, epochs[stage]):
                    pending.remove(job)
                    state[job] = "done"
                    log(f"{lr.run_name(stage, seed)}: already done")
                    continue
                if stage == "former":
                    ready = parasite_ready(seed)
                    if ready is False or state.get(("parasite", seed)) == "failed":
                        pending.remove(job)
                        state[job] = "blocked"
                        log(f"{lr.run_name(stage, seed)}: blocked (no parasite weights)")
                        continue
                    if ready is None:
                        continue
                pending.remove(job)
                plan = lr.plan_run(args.root, stage, seed, epochs[stage], args.epochs_to_test,
                                   args.opt, parasite_epochs=args.epochs)
                running[job] = (lr.start(plan, args.force), plan)
                started[job] = time.time()
                log(f"{plan['name']}: started")
            for job, (process, plan) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                stage, seed = job
                status = lr.finish(plan, code, lr.is_done(args.root, stage, seed, epochs[stage]))
                state[job] = status
                del running[job]
                log(f"{plan['name']}: {status} after {(time.time() - started[job]) / 60:.0f} min")
            time.sleep(5)
    except KeyboardInterrupt:
        log("interrupted, stopping runs")
        for process, plan in running.values():
            process.terminate()
        for (stage, seed), (process, plan) in running.items():
            process.wait()
            lr.finish(plan, 1, False)
        sys.exit(130)

    failed = [lr.run_name(*job) for job, st in state.items() if st in ("failed", "blocked")]
    log(f"finished: {sum(st == 'done' for st in state.values())} done, {len(failed)} not done")
    for name in failed:
        log(f"  not done: {name}")
    sys.exit(1 if failed else 0)
