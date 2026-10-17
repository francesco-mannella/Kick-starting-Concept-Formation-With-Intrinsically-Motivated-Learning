import atexit
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import time
from pathlib import Path

DEFAULT_DB = "localwandb.db"

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


def db_path(db=None):
    return Path(db or os.environ.get("LOCALWANDB_DB") or DEFAULT_DB).resolve()


def media_dir(db=None):
    path = db_path(db)
    return path.parent / f"{path.stem}_media"


def connect(db=None):
    path = db_path(db)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=60)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


class Image:
    def __init__(self, data, caption=None):
        self.path = Path(data)
        self.caption = caption


class Run:
    def __init__(self, con, run_id, media_root, name, project, entity, group, config):
        self._con = con
        self._media_root = media_root
        self._step = 0
        self.id = str(run_id)
        self._run_id = run_id
        self.name = name
        self.project = project
        self.entity = entity
        self.group = group
        self.config = config
        self.finished = False

    def log(self, data, step=None, commit=True):
        if step is None:
            step = self._step
        step = int(step)
        self._step = max(self._step, step + 1)

        metrics, media = [], []
        for key, value in data.items():
            if isinstance(value, Image):
                media.append((key, value))
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                print(f"localwandb: skipping non-numeric key {key!r}", file=sys.stderr)
                continue
            if math.isfinite(value):
                metrics.append((self._run_id, step, key, value))

        media_rows = []
        for key, image in media:
            safe_key = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
            rel = f"{self._run_id}/{step}_{safe_key}{image.path.suffix}"
            dest = self._media_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(image.path, dest)
            except OSError as e:
                print(f"localwandb: cannot store image {image.path}: {e}", file=sys.stderr)
                continue
            media_rows.append((self._run_id, step, key, rel))

        with self._con:
            self._con.executemany(
                "INSERT OR REPLACE INTO metrics VALUES (?, ?, ?, ?)", metrics
            )
            self._con.executemany(
                "INSERT OR REPLACE INTO media VALUES (?, ?, ?, ?)", media_rows
            )
            self._con.execute(
                "UPDATE runs SET updated = ? WHERE id = ?", (time.time(), self._run_id)
            )

    def finish(self, status="finished"):
        if self.finished:
            return
        self.finished = True
        with self._con:
            self._con.execute(
                "UPDATE runs SET status = ?, updated = ? WHERE id = ?",
                (status, time.time(), self._run_id),
            )
        self._con.close()


run = None
config = {}


def init(project=None, entity=None, name=None, group=None, config=None, db=None, **kwargs):
    global run
    con = connect(db)
    now = time.time()
    with con:
        cur = con.execute(
            "INSERT INTO runs (project, entity, name, grp, config, status, created, updated)"
            " VALUES (?, ?, ?, ?, ?, 'running', ?, ?)",
            (project, entity, name, group, json.dumps(config or {}, default=str), now, now),
        )
        run_id = cur.lastrowid
        if name is None:
            name = f"run-{run_id}"
            con.execute("UPDATE runs SET name = ? WHERE id = ?", (name, run_id))
    run = Run(con, run_id, media_dir(db), name, project, entity, group, config or {})
    globals()["config"] = run.config
    return run


def log(data, step=None, commit=True):
    if run is None:
        raise RuntimeError("localwandb.init() must be called before log()")
    run.log(data, step=step, commit=commit)


def finish():
    if run is not None:
        run.finish()


def _finish_at_exit():
    if run is not None and not run.finished:
        run.finish("crashed" if getattr(sys, "last_exc", None) is not None else "finished")


atexit.register(_finish_at_exit)
