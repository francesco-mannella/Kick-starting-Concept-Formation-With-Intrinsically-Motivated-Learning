#!/usr/bin/env python
import argparse
import json
import mimetypes
import sqlite3
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from localwandb import connect, db_path, media_dir

UI_FILE = Path(__file__).resolve().parent / "localwandb_ui.html"
STALE_AFTER = 900
MAX_POINTS = 300


def connect_ro(db=None):
    return sqlite3.connect(f"file:{db_path(db)}?mode=ro", uri=True, timeout=60)


def rows(con, sql, args=()):
    cur = con.execute(sql, args)
    names = [c[0] for c in cur.description]
    return [dict(zip(names, r)) for r in cur.fetchall()]


def list_runs(con):
    runs = rows(
        con,
        "SELECT r.id, r.project, r.entity, r.name, r.grp AS \"group\", r.status,"
        " r.created, r.updated,"
        " (SELECT MAX(step) FROM metrics WHERE run_id = r.id) AS last_step"
        " FROM runs r ORDER BY r.id DESC",
    )
    now = time.time()
    for r in runs:
        if r["status"] == "running" and now - r["updated"] > STALE_AFTER:
            r["status"] = "stale"
    return runs


KEYS_CACHE = {}


def run_keys(con, run_id):
    status = rows(con, "SELECT status FROM runs WHERE id = ?", (run_id,))
    final = bool(status) and status[0]["status"] != "running"
    if final and run_id in KEYS_CACHE:
        return KEYS_CACHE[run_id]
    metrics = [r["key"] for r in rows(
        con, "SELECT DISTINCT key FROM metrics WHERE run_id = ? ORDER BY key", (run_id,))]
    images = [r["key"] for r in rows(
        con, "SELECT DISTINCT key FROM media WHERE run_id = ? ORDER BY key", (run_id,))]
    found = {"metrics": metrics, "images": images}
    if final:
        KEYS_CACHE[run_id] = found
    return found


def run_config(con, run_id):
    found = rows(con, "SELECT config FROM runs WHERE id = ?", (run_id,))
    return json.loads(found[0]["config"]) if found else {}


def configs(con, run_ids):
    marks = ",".join("?" * len(run_ids))
    found = rows(con, f"SELECT id, config FROM runs WHERE id IN ({marks})", run_ids)
    return {r["id"]: json.loads(r["config"] or "{}") for r in found}


def decimate(pts, limit):
    if len(pts) <= limit:
        return pts
    stride = -(-len(pts) // limit)
    return [p for p in pts if p[0] % stride == 0 or p is pts[-1]]


def metrics(con, run_ids, keys):
    marks = ",".join("?" * len(keys))
    out = {}
    for run_id in run_ids:
        series = {}
        cur = con.execute(
            f"SELECT key, step, value FROM metrics WHERE run_id = ? AND key IN ({marks})",
            (run_id, *keys),
        )
        for key, step, value in cur:
            series.setdefault(key, []).append((step, value))
        out[run_id] = {k: decimate(v, MAX_POINTS) for k, v in series.items()}
    return out


def images(con, run_id, key):
    return rows(
        con,
        "SELECT step, path FROM media WHERE run_id = ? AND key = ? ORDER BY step",
        (run_id, key),
    )


FIGURE_TYPES = {".png": "image", ".jpg": "image", ".jpeg": "image", ".svg": "image",
                ".gif": "image", ".csv": "table", ".pdf": "other"}


def figures_dir(db, figures):
    return Path(figures) if figures else db_path(db).parent / "figures"


def list_figures(root):
    found = []
    if root.is_dir():
        for f in sorted(root.iterdir()):
            kind = FIGURE_TYPES.get(f.suffix.lower())
            if f.is_file() and kind:
                stat = f.stat()
                found.append({"name": f.name, "type": kind, "size": stat.st_size,
                              "mtime": stat.st_mtime})
    return found


class Handler(BaseHTTPRequestHandler):
    db = None
    figures = None

    def log_message(self, fmt, *args):
        pass

    def figures_root(self):
        return figures_dir(self.db, self.figures)

    def send_bytes(self, body, ctype, status=200):
        try:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_json(self, obj, status=200):
        self.send_bytes(json.dumps(obj).encode(), "application/json", status)

    def do_GET(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)
        path = unquote(url.path)

        if path in ("/", "/index.html"):
            self.send_bytes(UI_FILE.read_bytes(), "text/html; charset=utf-8")
            return

        for prefix, root in (("/media/", media_dir(self.db)), ("/figures/", self.figures_root())):
            if path.startswith(prefix):
                root = root.resolve()
                target = (root / path[len(prefix):]).resolve()
                if root not in target.parents or not target.is_file():
                    self.send_bytes(b"not found", "text/plain", 404)
                    return
                ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                self.send_bytes(target.read_bytes(), ctype)
                return

        if path == "/api/figures":
            self.send_json(list_figures(self.figures_root()))
            return

        try:
            con = connect_ro(self.db)
        except sqlite3.Error as e:
            self.send_json({"error": str(e)}, 503)
            return
        try:
            if path == "/api/runs":
                self.send_json(list_runs(con))
            elif path == "/api/keys":
                run_id = int(query["run"][0])
                self.send_json(run_keys(con, run_id))
            elif path == "/api/config":
                run_id = int(query["run"][0])
                self.send_json(run_config(con, run_id))
            elif path == "/api/configs":
                run_ids = [int(x) for x in query["runs"][0].split(",") if x]
                self.send_json(configs(con, run_ids) if run_ids else {})
            elif path == "/api/metrics":
                run_ids = [int(x) for x in query["runs"][0].split(",") if x]
                keys = [k for k in query["keys"][0].split(",") if k]
                self.send_json(metrics(con, run_ids, keys))
            elif path == "/api/images":
                run_id = int(query["run"][0])
                found = images(con, run_id, query["key"][0])
                self.send_json([{"step": r["step"], "url": f"/media/{r['path']}"} for r in found])
            else:
                self.send_bytes(b"not found", "text/plain", 404)
        except (KeyError, ValueError, IndexError):
            self.send_bytes(b"bad request", "text/plain", 400)
        except sqlite3.Error as e:
            self.send_json({"error": str(e)}, 503)
        finally:
            con.close()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Local simulation data viewer")
    parser.add_argument("--db", default=None, help="Path of the database file")
    parser.add_argument("--figures", default=None,
                        help="Folder of figures and tables to show (default: <db folder>/figures)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    Handler.db = args.db
    Handler.figures = args.figures
    connect(args.db).close()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Database: {db_path(args.db)}")
    print(f"Figures: {figures_dir(args.db, args.figures)}")
    print(f"Serving on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
