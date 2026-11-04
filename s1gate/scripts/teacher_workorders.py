"""Turn configs/synth.pos.yaml into teacher work orders (one per agent chunk).

Each chunk is a list of batches; each batch = (scenario or pair id, row count, style card).
Style cards rotate deterministically through the label-neutral axes and, for scenarios that
allow it, the format-noise axes, so different agents cannot converge on one style.

    python -I scripts/teacher_workorders.py configs/synth.pos.yaml OUT.json [--max-rows 150] [--pairs 25]
"""
import argparse
import json
import random

import yaml

ap = argparse.ArgumentParser()
ap.add_argument("recipe")
ap.add_argument("out")
ap.add_argument("--max-rows", type=int, default=150)
ap.add_argument("--pairs", type=int, default=25, help="pairs per pair type")
ap.add_argument("--seed", type=int, default=11)
args = ap.parse_args()

R = yaml.safe_load(open(args.recipe, encoding="utf-8"))
rng = random.Random(args.seed)
var, noise = R["variation"], R["noise"]
B = R.get("batch_size", 10)


def card(idx, noise_mode):
    c = {axis: values[(idx * (i + 3) + rng.randrange(len(values))) % len(values)]
         for i, (axis, values) in enumerate(var.items())}
    if noise_mode == "required":
        axes = rng.sample(list(noise), k=rng.choice([1, 1, 2]))
        c["noise"] = [rng.choice(noise[a]) for a in axes]
    elif noise_mode == "optional":
        k = rng.choice([0, 0, 1, 2])
        c["noise"] = [rng.choice(noise[a]) for a in rng.sample(list(noise), k=k)] if k else ["none"]
    else:
        c["noise"] = ["none - canonical formats only"]
    return c


def yaml_target(t):
    return {k: str(v) if not isinstance(v, bool) else ("yes" if v else "no") for k, v in t.items()}


units = []  # (kind, id, rows, spec)
for s in R["scenarios"]:
    n = max(1, round(R["n_per_scenario"] * s.get("weight", 1.0)))
    units.append(("scenario", s["id"], n, {"target": yaml_target(s["target"]), "hint": s["hint"],
                                         "noise": s.get("noise", "optional")}))
for p in R.get("pairs", []):
    units.append(("pair", p["id"], 2 * args.pairs, {"change": p["change"], "a": yaml_target(p["a"]),
                                                   "b": yaml_target(p["b"])}))

chunks, cur, cur_rows = [], [], 0
for u in units:
    if cur and (cur_rows + u[2] > args.max_rows or cur[0][0] != u[0]):
        chunks.append(cur)
        cur, cur_rows = [], 0
    cur.append(u)
    cur_rows += u[2]
if cur:
    chunks.append(cur)

orders, gidx = [], 0
for ci, chunk in enumerate(chunks):
    batches = []
    for kind, uid, n, spec in chunk:
        left = n
        while left > 0:
            k = min(B, left)
            mode = spec.get("noise", "optional") if kind == "scenario" else "optional"
            batches.append({"kind": kind, "id": uid, "rows": k, "style": card(gidx, mode)})
            gidx += 1
            left -= k
    ids = sorted({u[1] for u in chunk})
    orders.append({"chunk": f"t{ci:02d}", "kind": chunk[0][0], "units": [
        {"kind": k, "id": i, "rows": n, **spec} for k, i, n, spec in chunk],
        "batches": batches, "rows": sum(b["rows"] for b in batches), "ids": ids})

json.dump(orders, open(args.out, "w"), indent=1, ensure_ascii=False)
print(f"{len(orders)} chunks, {sum(o['rows'] for o in orders)} rows")
for o in orders:
    print(o["chunk"], o["kind"], o["rows"], ",".join(o["ids"]))
