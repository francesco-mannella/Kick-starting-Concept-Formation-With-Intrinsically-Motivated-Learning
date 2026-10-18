#!/usr/bin/env python
import argparse
import json
import sys
import time

import launch_run as lr


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Launch independent runs: conditions (name -> list of KEY=VALUE options) x seeds"
    )
    parser.add_argument("--conditions", required=True, help="JSON file {name: [KEY=VALUE, ...]}")
    parser.add_argument("--only", default=None, help="Comma separated condition names to run")
    parser.add_argument("--root", required=True)
    parser.add_argument("--db", default=None, help="Database file (default: <root>/runs.db)")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("-p", "--parallel", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--epochs-to-test", type=int, default=10)
    parser.add_argument("--force", action="store_true", help="Delete incomplete run folders")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


if __name__ == "__main__":
    args = parse_arguments()
    conditions = json.load(open(args.conditions))
    if args.only:
        conditions = {k: v for k, v in conditions.items() if k in args.only.split(",")}
    seeds = [int(s) for s in args.seeds.split(",")]
    jobs = [(name, seed, options) for name, options in conditions.items() for seed in seeds]

    if args.dry_run:
        for name, seed, options in jobs:
            plan = lr.plan_run(args.root, "agent", seed, args.epochs, args.epochs_to_test, options,
                               tag=name, db=args.db)
            print(f"cd {plan['cwd']} && {' '.join(plan['command'])}")
        sys.exit(0)

    running, started, state = {}, {}, {}
    log(f"{len(jobs)} runs, {args.parallel} in parallel, root {args.root}")
    try:
        while jobs or running:
            while jobs and len(running) < args.parallel:
                name, seed, options = jobs.pop(0)
                if lr.is_done(args.root, "agent", seed, args.epochs, name):
                    state[(name, seed)] = "done"
                    log(f"{lr.run_name('agent', seed, name)}: already done")
                    continue
                plan = lr.plan_run(args.root, "agent", seed, args.epochs, args.epochs_to_test,
                                   options, tag=name, db=args.db)
                running[(name, seed)] = (lr.start(plan, args.force), plan)
                started[(name, seed)] = time.time()
                log(f"{plan['name']}: started")
            for key, (process, plan) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                name, seed = key
                status = lr.finish(plan, code, lr.is_done(args.root, "agent", seed, args.epochs, name))
                state[key] = status
                del running[key]
                log(f"{plan['name']}: {status} after {(time.time() - started[key]) / 60:.0f} min")
            time.sleep(5)
    except KeyboardInterrupt:
        log("interrupted, stopping runs")
        for process, plan in running.values():
            process.terminate()
        for process, plan in running.values():
            process.wait()
            lr.finish(plan, 1, False)
        sys.exit(130)

    failed = [lr.run_name("agent", seed, name) for (name, seed), st in state.items() if st != "done"]
    log(f"finished: {sum(st == 'done' for st in state.values())} done, {len(failed)} not done")
    for name in failed:
        log(f"  not done: {name}")
    sys.exit(1 if failed else 0)
