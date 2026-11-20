import argparse
import json
import math
import re
import sqlite3
from pathlib import Path

parser = argparse.ArgumentParser(description="Insert recomputed mi2 values (mi2_recompute.py output) into a sweep database")
parser.add_argument("--db", required=True)
parser.add_argument("--results", required=True, help="Folder with <run>_<epoch>_<agent|par>.json files")
parser.add_argument("--no-backup", action="store_true")
args = parser.parse_args()

db = Path(args.db)
if not args.no_backup:
    backup = db.with_name(db.name + ".bak_before_mi2")
    if not backup.exists():
        src, dst = sqlite3.connect(db), sqlite3.connect(backup)
        src.backup(dst)
        dst.close()
        src.close()
        print("backup written", backup)
con = sqlite3.connect(db)
ids = dict(con.execute("select name, id from runs"))
rows, skipped = [], 0
for f in sorted(Path(args.results).glob("*.json")):
    run, epoch, _series = re.match(r"(.+)_(\d+)_(agent|par)\.json", f.name).groups()
    for key, value in json.load(open(f)).items():
        if value is None or math.isnan(value):
            skipped += 1
            continue
        rows.append((ids[run], int(epoch), key, value))
con.executemany("INSERT OR REPLACE INTO metrics (run_id, step, key, value) VALUES (?,?,?,?)", rows)
con.commit()
print(f"inserted {len(rows)} rows, skipped {skipped} NaN (too few segments)")
print(con.execute("select count(distinct key), count(*) from metrics where key like 'mi2_%'").fetchone())
