"""Tag each teacher row with how it was authored, from the teacher agents' own reports.

  hand   - the agent chose every row's values itself (Opus chunks t*, and every h* chunk)
  script - the agent wrote a seeded generator script that produced the rows (Sonnet t* chunks)

Script rows are valid and blind-verified, but less varied; the dataset card reports the split
so they can be down-weighted or dropped in an ablation.

    python -I scripts/tag_authoring.py data/synth/teacher.verified.jsonl data/synth/teacher.rejected.jsonl
"""
import json
import sys
from pathlib import Path

for path in sys.argv[1:]:
    p = Path(path)
    if not p.exists():
        continue
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in rows:
        m = r.setdefault("meta", {})
        if "authoring" not in m:
            m["authoring"] = "script" if m.get("teacher") == "sonnet" and r["id"].startswith("t") else "hand"
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{p}: {sum(r['meta']['authoring'] == 'hand' for r in rows)} hand, "
          f"{sum(r['meta']['authoring'] == 'script' for r in rows)} script")
