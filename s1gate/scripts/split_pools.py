"""Split the invented company pools into disjoint train and eval partitions.

Entities (distributors, customers) are assigned by a stable hash of their name, per region:
the lowest-hash 3 distributors and ~30% of customers go to eval. Train-side generators never
see eval companies, so `s1gate clean` decontamination measures generalisation, not names.

    python -I scripts/split_pools.py data/pools/raw/companies_a.json data/pools/raw/companies_b.json --out data/pools
"""
import argparse
import hashlib
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--out", required=True)
ap.add_argument("--eval-distributors", type=int, default=3)
ap.add_argument("--eval-customer-frac", type=float, default=0.3)
args = ap.parse_args()


def h(name):
    return hashlib.sha1(name.strip().lower().encode()).hexdigest()


merged = {}
for f in args.files:
    for region, block in json.load(open(f, encoding="utf-8")).items():
        merged.setdefault(region, {"distributors": [], "customers": []})
        merged[region]["distributors"] += block["distributors"]
        merged[region]["customers"] += block["customers"]

train, evals, seen = {}, {}, set()
for region, block in sorted(merged.items()):
    ds = [d for d in block["distributors"] if d["name"].strip().lower() not in seen]
    seen |= {d["name"].strip().lower() for d in ds}
    cs = [c for c in block["customers"] if c["name"].strip().lower() not in seen]
    seen |= {c["name"].strip().lower() for c in cs}
    ds.sort(key=lambda d: h(d["name"]))
    cs.sort(key=lambda c: h(c["name"]))
    k = round(len(cs) * args.eval_customer_frac)
    evals[region] = {"distributors": ds[:args.eval_distributors], "customers": cs[:k]}
    train[region] = {"distributors": ds[args.eval_distributors:], "customers": cs[k:]}

out = Path(args.out)
for name, part in (("train", train), ("eval", evals)):
    (out / name).mkdir(parents=True, exist_ok=True)
    (out / name / "companies.json").write_text(json.dumps(part, indent=1, ensure_ascii=False))
    nd = sum(len(v["distributors"]) for v in part.values())
    nc = sum(len(v["customers"]) for v in part.values())
    print(f"{name}: {len(part)} regions, {nd} distributors, {nc} customers")
