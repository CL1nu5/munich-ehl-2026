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
OUT = ROOT / "results" / "dashboard.html"
SLOTS = {
    "/*__DASHBOARD_DATA__*/": ROOT / "results" / "dashboard_data.json",
    "/*__FRONTIER_DATA__*/": ROOT / "results" / "frontier_data.json",
}

def main():
    html = TPL.read_text(encoding="utf-8")
    for token, path in SLOTS.items():
        if not path.exists():
            sys.exit(f"missing {path} — run scripts/build_dashboard_data.py "
                     f"and scripts/build_frontier.py first")
        if token not in html:
            sys.exit(f"placeholder {token} not found in {TPL}")
        payload = json.dumps(json.loads(path.read_text(encoding="utf-8")), separators=(",", ":"))
        # </script> inside the JSON would close the host <script> tag early
        html = html.replace(token, payload.replace("</", "<\\/"))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size:,} bytes)")

if __name__ == "__main__":
    main()
