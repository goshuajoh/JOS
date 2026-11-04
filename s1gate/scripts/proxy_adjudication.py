"""Proxy dev/eval labels: author + two blind annotators + rule audit -> accepted / to adjudicate.

    prep:   python -I scripts/proxy_adjudication.py prep --raw data/proxy/raw/*.jsonl \
                --a data/proxy/blind/*.A.labels.jsonl --b data/proxy/blind/*.B.labels.jsonl \
                --keys data/proxy/blind/key_r*.json --out-dir data/proxy
            -> data/proxy/unanimous.jsonl, data/proxy/disputed.jsonl, data/proxy/agreement.json
    final:  python -I scripts/proxy_adjudication.py final --out-dir data/proxy \
                --adjudicated data/proxy/adjudicated.jsonl
            -> data/proxy/labelled.jsonl (unanimous + adjudicated, dropped rows removed)

A row is unanimous when the author and both annotators give the same three answers and the
rule audit finds no violation. Everything else goes to adjudication with all three opinions.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from s1gate.agreement import cohen_kappa  # noqa: E402
from s1gate.domains import pos  # noqa: E402

Q = ["row_action", "needs_human", "severity"]


def read(paths):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    return rows


def labels_by_id(paths, key=None):
    key = key or {}
    return {key.get(str(r["id"]), str(r["id"])): {q: str(r.get(q)) for q in Q} for r in read(paths)}


def pairwise(a: dict, b: dict) -> dict:
    out = {}
    for q in Q:
        pairs = [(a[i][q], b[i][q]) for i in a if i in b]
        n = len(pairs)
        out[q] = {"n": n, "agreement": round(sum(x == y for x, y in pairs) / n, 4) if n else None,
                  "kappa": round(cohen_kappa(pairs), 4) if n else None,
                  "top_disagreements": [f"{x}->{y} x{c}" for (x, y), c in
                                        Counter((x, y) for x, y in pairs if x != y).most_common(6)]}
    return out


def prep(args):
    raw = read(args.raw)
    key = {}
    for k in args.keys or []:
        key.update(json.loads(Path(k).read_text()))
    A, B = labels_by_id(args.a, key), labels_by_id(args.b, key)
    catalogue = pos.load_catalogue(str(ROOT / "configs/pools/parts.yaml"))
    author = {r["id"]: {q: r["labels"][q] for q in Q} for r in raw}
    out = Path(args.out_dir)
    unanimous, disputed = [], []
    for r in raw:
        i = r["id"]
        a, b, au = A.get(i), B.get(i), author[i]
        audit = [f.message for f in pos.check_labels(r["text"], au, catalogue) if f.level == "violation"]
        if a == au and b == au and not audit:
            unanimous.append({**r, "meta": {**r["meta"], "label_status": "unanimous"}})
        else:
            disputed.append({"id": i, "text": r["text"], "author": au, "annotator_a": a, "annotator_b": b,
                             "audit_violations_of_author_labels": audit, "meta": r["meta"]})
    with open(out / "unanimous.jsonl", "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in unanimous)
    with open(out / "disputed.jsonl", "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in disputed)
    report = {"rows": len(raw), "unanimous": len(unanimous), "disputed": len(disputed),
              "missing_a": sum(1 for r in raw if r["id"] not in A),
              "missing_b": sum(1 for r in raw if r["id"] not in B),
              "a_vs_b": pairwise(A, B), "author_vs_a": pairwise(author, A), "author_vs_b": pairwise(author, B)}
    (out / "agreement.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, dict)}, indent=1))
    for k in ("a_vs_b", "author_vs_a", "author_vs_b"):
        print(k, {q: (v["agreement"], v["kappa"]) for q, v in report[k].items()})


def final(args):
    out = Path(args.out_dir)
    unanimous = read([out / "unanimous.jsonl"])
    disputed = {r["id"]: r for r in read([out / "disputed.jsonl"])}
    adjudicated = {str(r["id"]): r for r in read(args.adjudicated)}
    rows, dropped, changed = list(unanimous), [], 0
    for i, d in disputed.items():
        adj = adjudicated.get(i)
        if adj is None or adj.get("drop"):
            dropped.append({"id": i, "reason": (adj or {}).get("reason", "not adjudicated")})
            continue
        labels = {q: str(adj[q]) for q in Q}
        changed += labels != d["author"]
        rows.append({"id": i, "text": d["text"], "labels": labels, "source": "proxy",
                     "meta": {**d["meta"], "label_status": "adjudicated", "author_labels": d["author"],
                              "annotator_a": d["annotator_a"], "annotator_b": d["annotator_b"],
                              "adjudication_note": adj.get("reason", "")}})
    rows.sort(key=lambda r: r["id"])
    with open(out / "labelled.jsonl", "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
    (out / "dropped.json").write_text(json.dumps(dropped, indent=1))
    print(f"{len(rows)} labelled ({len(unanimous)} unanimous, {len(rows) - len(unanimous)} adjudicated, "
          f"{changed} changed from the author's labels), {len(dropped)} dropped")


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
p = sub.add_parser("prep")
p.add_argument("--raw", nargs="+", required=True)
p.add_argument("--a", nargs="+", required=True)
p.add_argument("--b", nargs="+", required=True)
p.add_argument("--keys", nargs="*", help="opaque id -> row id files from blind-export --key-out")
p.add_argument("--out-dir", required=True)
p = sub.add_parser("final")
p.add_argument("--out-dir", required=True)
p.add_argument("--adjudicated", nargs="+", required=True)
args = ap.parse_args()
prep(args) if args.cmd == "prep" else final(args)
