#!/usr/bin/env python
import argparse
import shutil
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project TEXT, entity TEXT, name TEXT, grp TEXT,
    config TEXT, status TEXT, created REAL, updated REAL
);
CREATE TABLE IF NOT EXISTS metrics (
    run_id INTEGER, step INTEGER, key TEXT, value REAL,
    PRIMARY KEY (run_id, step, key)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS media (
    run_id INTEGER, step INTEGER, key TEXT, path TEXT,
    PRIMARY KEY (run_id, step, key)
) WITHOUT ROWID;
"""
OVERLAP = 5


def media_root(db):
    db = Path(db)
    return db.parent / f"{db.stem}_media"


def merge(out, sources):
    target = sqlite3.connect(out)
    target.executescript(SCHEMA)
    running = 0
    for label, path in sources:
        if not Path(path).is_file():
            continue
        source = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        for rid, project, entity, name, grp, config, status, created, updated in source.execute(
            "SELECT id, project, entity, name, grp, config, status, created, updated FROM runs"
        ).fetchall():
            new_name = f"{label}_{name}"
            row = target.execute("SELECT id FROM runs WHERE name = ?", (new_name,)).fetchone()
            if row is None:
                cursor = target.execute(
                    "INSERT INTO runs (project, entity, name, grp, config, status, created, updated) VALUES (?,?,?,?,?,?,?,?)",
                    (project, entity, new_name, f"{label}_{grp}", config, status, created, updated))
                target_id = cursor.lastrowid
            else:
                target_id = row[0]
                target.execute("UPDATE runs SET status = ?, updated = ? WHERE id = ?", (status, updated, target_id))
            last = target.execute("SELECT COALESCE(MAX(step), -1) FROM metrics WHERE run_id = ?", (target_id,)).fetchone()[0]
            rows = source.execute("SELECT step, key, value FROM metrics WHERE run_id = ? AND step > ?",
                                  (rid, last - OVERLAP)).fetchall()
            target.executemany("INSERT OR REPLACE INTO metrics (run_id, step, key, value) VALUES (?,?,?,?)",
                               [(target_id, s, k, v) for s, k, v in rows])
            for step, key, rel in source.execute("SELECT step, key, path FROM media WHERE run_id = ?", (rid,)).fetchall():
                new_rel = f"{label}/{rel}"
                origin, dest = media_root(path) / rel, media_root(out) / new_rel
                if not dest.exists() and origin.is_file():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(origin, dest)
                target.execute("INSERT OR IGNORE INTO media (run_id, step, key, path) VALUES (?,?,?,?)",
                               (target_id, step, key, new_rel))
            running += status == "running"
        source.close()
    target.commit()
    target.close()
    return running


def parse_arguments():
    parser = argparse.ArgumentParser(description="Merge several local run databases into one, prefixing run names and groups")
    parser.add_argument("--source", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--out", required=True)
    parser.add_argument("--watch", type=int, default=0, help="Repeat every N seconds until no source run is running")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    sources = [tuple(item.split("=", 1)) for item in args.source]
    while True:
        running = merge(args.out, sources)
        print(time.strftime("%H:%M:%S"), f"merged; {running} source runs still running", flush=True)
        if not args.watch:
            break
        if running == 0:
            break
        time.sleep(args.watch)
