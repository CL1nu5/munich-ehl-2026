#!/usr/bin/env python3
"""Inject results/dashboard_data.json into dashboard/template.html -> results/dashboard.html

Full refresh after new results land:
    python scripts/build_dashboard_data.py
    python dashboard/build.py
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TPL = ROOT / "dashboard" / "template.html"
DATA = ROOT / "results" / "dashboard_data.json"
OUT = ROOT / "results" / "dashboard.html"
TOKEN = "/*__DASHBOARD_DATA__*/"

def main():
    if not DATA.exists():
        sys.exit(f"missing {DATA} — run scripts/build_dashboard_data.py first")
    html = TPL.read_text(encoding="utf-8")
    if TOKEN not in html:
        sys.exit(f"placeholder {TOKEN} not found in {TPL}")
    payload = json.dumps(json.loads(DATA.read_text(encoding="utf-8")), separators=(",", ":"))
    # </script> inside the JSON would close the host <script> tag early
    payload = payload.replace("</", "<\\/")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html.replace(TOKEN, payload), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size:,} bytes)")

if __name__ == "__main__":
    main()
