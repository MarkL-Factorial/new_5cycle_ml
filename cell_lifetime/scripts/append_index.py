#!/usr/bin/env python
"""Append a row to cell_lifetime/INDEX.md's "Phase log" table.

Used by run_routine.sh (cloud routines) and monday_smoke_real_data.sh
(local) to keep the INDEX append-only across both surfaces.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--ts", required=True, help="UTC timestamp, e.g. 20260516T140000Z")
    ap.add_argument("--status", required=True, choices=["OK", "BROKEN", "BLOCKED"])
    ap.add_argument("--n-pass", type=int, default=0)
    ap.add_argument("--n-fail", type=int, default=0)
    ap.add_argument("--summary", default="")
    ap.add_argument("--commit", default="(pending)")
    ap.add_argument("--files-added", default="(auto)")
    ap.add_argument("--surface", default="cloud")
    args = ap.parse_args()

    index_path = (
        Path(__file__).resolve().parents[1] / "INDEX.md"
    )
    if not index_path.exists():
        print(f"WARN: {index_path} not found; skipping append")
        return 0

    txt = index_path.read_text()
    ended = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    n_total = args.n_pass + args.n_fail
    row = (
        f"| {args.phase} | {args.surface} | {args.ts} | {ended} | {args.status} | "
        f"{args.commit} | {args.files_added} | {n_total} | {args.n_pass} | {args.summary} |"
    )

    # Find the Phase log table and append after its existing rows
    marker = "## Phase log (append-only)"
    if marker not in txt:
        print(f"WARN: marker {marker!r} missing in INDEX.md; appending at EOF")
        new_txt = txt.rstrip() + "\n\n" + row + "\n"
    else:
        lines = txt.splitlines()
        out: list[str] = []
        in_table = False
        appended = False
        for i, line in enumerate(lines):
            out.append(line)
            if line.startswith(marker):
                in_table = True
                continue
            if in_table and line.startswith("|") and not line.startswith("|---"):
                # Track the last table row; we'll append after the table ends
                pass
            if in_table and not appended and (line.strip() == "" or line.startswith("## ")):
                # End of table — append before this blank line / next header
                # Strip the just-pushed line first, insert row, then re-push.
                out.pop()
                out.append(row)
                out.append(line)
                appended = True
                in_table = False
        if not appended:
            out.append(row)
        new_txt = "\n".join(out) + "\n"

    # Update Last updated line
    today = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    new_lines: list[str] = []
    for line in new_txt.splitlines():
        if line.startswith("Last updated:"):
            new_lines.append(f"Last updated: {today}")
        else:
            new_lines.append(line)
    index_path.write_text("\n".join(new_lines) + "\n")
    print(f"[append_index] added row for {args.phase} status={args.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
