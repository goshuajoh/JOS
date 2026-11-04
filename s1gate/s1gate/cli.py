"""s1gate command line. One subcommand per step; run `s1gate <cmd> -h` for options.

  1  validate, export-schema, label, split
  2  eval
  3  synth, clean
  4-5 train-gliner, train-qwen
  6  calibrate
  7  thresholds
  8  serve
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from .records import (Record, decontaminate, dedupe, label_counts, read_jsonl, split,
                      validate_records, write_jsonl)
from .schema import DecisionSchema


# ---------------------------------------------------------------- helpers

def _schema(args) -> DecisionSchema:
    return DecisionSchema.load(args.schema)


def _write_json(path: str, data) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _read_preds(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _backend(args):
    from .backends import make_backend
    kw = {}
    if getattr(args, "batch_size", None) and args.backend in ("gliner", "qwen"):
        kw["batch_size"] = args.batch_size
    return make_backend(args.backend, model=getattr(args, "model", None),
                        adapter=getattr(args, "adapter", None), url=getattr(args, "url", None),
                        device=getattr(args, "device", None), **kw)


def _fmt(v, digits=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        return "nan" if v != v else f"{v:.{digits}f}"
    return str(v)


# ---------------------------------------------------------------- step 1

def cmd_validate(args) -> int:
    schema = _schema(args)
    print(f"schema {schema.name} v{schema.version} fingerprint {schema.fingerprint()}")
    bad = 0
    for path in args.data or []:
        records = read_jsonl(path)
        errors = validate_records(records, schema)
        bad += len(errors)
        print(f"\n{path}: {len(records)} records, {len(errors)} problems")
        for e in errors[:50]:
            print("  ", e)
        for q, counts in label_counts(records, schema).items():
            total = sum(counts.values())
            dist = ", ".join(f"{o}={counts.get(o, 0)}" for o in schema.question(q).labels)
            print(f"  {q} ({total} labelled): {dist}")
    return 1 if bad else 0


def cmd_export_schema(args) -> int:
    schema = _schema(args)
    _write_json(args.out, schema.json_schema())
    print(f"wrote {args.out}")
    return 0


def cmd_label(args) -> int:
    """Terminal labelling: number keys pick an option, s skips, q saves and quits."""
    schema = _schema(args)
    records = read_jsonl(args.data)
    done = {r.id: r for r in read_jsonl(args.out)} if Path(args.out).exists() else {}
    try:
        for r in records:
            current = done.get(r.id, r)
            missing = [q for q in schema.questions if q.name not in current.labels]
            if not missing:
                continue
            print("\n" + "=" * 80 + f"\n{r.id}\n{r.text}\n")
            for q in missing:
                print(f"{q.prompt}")
                for i, (o, d) in enumerate(q.options.items(), 1):
                    print(f"  {i}. {o}" + (f"  - {d}" if d else ""))
                choice = input("> ").strip().lower()
                if choice == "q":
                    raise KeyboardInterrupt
                if choice.isdigit() and 1 <= int(choice) <= len(q.options):
                    current.labels[q.name] = q.labels[int(choice) - 1]
            done[r.id] = current
    except (KeyboardInterrupt, EOFError):
        print("\nstopping")
    write_jsonl(args.out, done.values())
    print(f"saved {len(done)} records to {args.out}")
    return 0


def cmd_split(args) -> int:
    records = read_jsonl(args.data)
    rest, held = split(records, args.frac, args.seed, args.stratify)
    write_jsonl(args.out_rest, rest)
    write_jsonl(args.out_held, held)
    print(f"{len(rest)} -> {args.out_rest}, {len(held)} -> {args.out_held}")
    return 0


# ---------------------------------------------------------------- step 2

def cmd_eval(args) -> int:
    from .calibrate import apply_temperature, load_calibration
    from .metrics import evaluate_question

    schema = _schema(args)
    records = read_jsonl(args.data)
    errors = validate_records(records, schema)
    if errors:
        print("\n".join(errors[:20]), file=sys.stderr)
        return 1
    backend = _backend(args)
    temps = load_calibration(args.calibration, schema)

    preds, latencies = [], []
    bs = args.batch_size or 16
    for i in range(0, len(records), bs):
        chunk = records[i:i + bs]
        t0 = time.perf_counter()
        out = backend.predict_batch([r.text for r in chunk], schema)
        per = (time.perf_counter() - t0) * 1000 / len(chunk)
        for r, probs in zip(chunk, out):
            latencies.append(per)
            preds.append({"id": r.id, "probs": probs, "gold": r.labels, "latency_ms": round(per, 2)})

    if args.preds_out:
        Path(args.preds_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.preds_out, "w", encoding="utf-8") as fh:
            for p in preds:
                fh.write(json.dumps(p) + "\n")

    reports = []
    for q in schema.questions:
        rows = [p for p in preds if q.name in p["gold"]]
        probs = [apply_temperature(p["probs"][q.name], temps.get(q.name, 1.0)) for p in rows]
        reports.append(evaluate_question(q.name, probs, [p["gold"][q.name] for p in rows],
                                         q.labels, ordinal=q.kind == "score"))

    p50 = statistics.median(latencies) if latencies else float("nan")
    print(f"\nbackend={backend.name} records={len(records)} calibrated={bool(temps)} "
          f"latency p50={p50:.1f} ms/record")
    header = f"{'question':<16}{'n':>5}{'acc':>8}{'macroF1':>9}{'NLL':>8}{'Brier':>8}{'ECE':>8}{'conf':>8}{'MAE':>7}"
    print(header + "\n" + "-" * len(header))
    for rep in reports:
        print(f"{rep.question:<16}{rep.n:>5}{_fmt(rep.accuracy):>8}{_fmt(rep.macro_f1):>9}"
              f"{_fmt(rep.nll):>8}{_fmt(rep.brier):>8}{_fmt(rep.ece):>8}"
              f"{_fmt(rep.mean_confidence):>8}{_fmt(rep.mae, 2):>7}")
    if args.report_out:
        _write_json(args.report_out, {"backend": backend.name, "data": args.data,
                                      "latency_p50_ms": p50, "calibrated": bool(temps),
                                      "questions": [r.to_dict() for r in reports]})
        print(f"\nreport -> {args.report_out}")
    return 0


# ---------------------------------------------------------------- step 3

def cmd_synth(args) -> int:
    from .synth import Recipe, generate, make_llm

    schema = _schema(args)
    recipe = Recipe.load(args.recipe)
    llm = make_llm(args.provider, args.model, args.base_url)
    records = generate(schema, recipe, llm, seed=args.seed, verify=not args.no_verify, scale=args.scale)
    n = write_jsonl(args.out, records)
    print(f"wrote {n} synthetic records to {args.out}")
    return 0


def cmd_clean(args) -> int:
    records: list[Record] = []
    for path in args.train:
        records.extend(read_jsonl(path))
    kept, dups = dedupe(records, args.dup)
    held: list[Record] = []
    for path in args.held_out or []:
        held.extend(read_jsonl(path))
    kept, leaks = decontaminate(kept, held, args.contam) if held else (kept, [])
    write_jsonl(args.out, kept)
    print(f"{len(records)} in, {len(dups)} near-duplicates, {len(leaks)} too close to held-out, "
          f"{len(kept)} -> {args.out}")
    return 0


# ---------------------------------------------------------------- steps 4-5

def _train_inputs(args):
    schema = _schema(args)
    train = []
    for path in args.train:
        train.extend(read_jsonl(path))
    dev = read_jsonl(args.dev) if args.dev else None
    return schema, train, dev


def cmd_train_gliner(args) -> int:
    from .train_gliner import train

    schema, tr, dev = _train_inputs(args)
    path = train(tr, dev, schema, args.out, base=args.base, epochs=args.epochs,
                 batch_size=args.batch_size, encoder_lr=args.encoder_lr, task_lr=args.task_lr,
                 use_lora=args.lora, augment=args.augment)
    print(f"model -> {path}  (evaluate with: s1gate eval --backend gliner --model {path} ...)")
    return 0


def cmd_train_qwen(args) -> int:
    from .train_qwen import train

    schema, tr, dev = _train_inputs(args)
    path = train(tr, dev, schema, args.out, base=args.base, epochs=args.epochs,
                 batch_size=args.batch_size, grad_accum=args.grad_accum, lr=args.lr,
                 lora_r=args.lora_r)
    print(f"adapter -> {path}  (evaluate with: s1gate eval --backend qwen --model {args.base} --adapter {path} ...)")
    return 0


# ---------------------------------------------------------------- steps 6-7

def cmd_calibrate(args) -> int:
    from .calibrate import fit_all

    schema = _schema(args)
    result = fit_all(schema, _read_preds(args.preds))
    _write_json(args.out, result)
    for q, t in result["temperatures"].items():
        print(f"{q:<16} T={t:<8} {result['detail'][q]}")
    print(f"-> {args.out}")
    return 0


def cmd_thresholds(args) -> int:
    from .calibrate import load_calibration
    from .thresholds import fit_all

    schema = _schema(args)
    targets: float | dict[str, float] = args.target
    if args.target_q:
        targets = {q.name: args.target for q in schema.questions}
        for item in args.target_q:
            name, value = item.split("=")
            targets[name] = float(value)
    result = fit_all(schema, _read_preds(args.preds), load_calibration(args.calibration, schema),
                     targets, args.min_support)
    _write_json(args.out, result)
    for q, info in result["thresholds"].items():
        print(f"{q:<16} {info}")
    print(f"-> {args.out}")
    return 0


# ---------------------------------------------------------------- step 8

def cmd_serve(args) -> int:
    import uvicorn

    from .gate import Gate
    from .serve import create_app

    gate = Gate.from_files(args.schema, _backend(args), args.calibration, args.thresholds, args.log)
    uvicorn.run(create_app(gate, args.corrections), host=args.host, port=args.port)
    return 0


# ---------------------------------------------------------------- parser

def _add_backend_args(p) -> None:
    p.add_argument("--backend", default="gliner", choices=["keyword", "gliner", "qwen", "endpoint"])
    p.add_argument("--model", help="HF id or local path (GLiNER checkpoint or Qwen base)")
    p.add_argument("--adapter", help="LoRA adapter dir (qwen backend)")
    p.add_argument("--url", help="base URL (endpoint backend)")
    p.add_argument("--device", help="cpu, cuda, cuda:0, mps")
    p.add_argument("--batch-size", type=int, default=16)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="s1gate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("validate", help="check a schema and label files")
    p.add_argument("--schema", required=True)
    p.add_argument("--data", nargs="*")
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("export-schema", help="write the JSON schema of one gate answer")
    p.add_argument("--schema", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_export_schema)

    p = sub.add_parser("label", help="label records in the terminal")
    p.add_argument("--schema", required=True)
    p.add_argument("--data", required=True, help="unlabelled or partly labelled JSONL")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_label)

    p = sub.add_parser("split", help="split one JSONL file in two")
    p.add_argument("--data", required=True)
    p.add_argument("--frac", type=float, default=0.3, help="share that goes to --out-held")
    p.add_argument("--stratify", help="question whose label keeps proportions")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--out-rest", required=True)
    p.add_argument("--out-held", required=True)
    p.set_defaults(fn=cmd_split)

    p = sub.add_parser("eval", help="score a backend on labelled data")
    p.add_argument("--schema", required=True)
    p.add_argument("--data", required=True)
    _add_backend_args(p)
    p.add_argument("--calibration", help="apply temperatures before scoring")
    p.add_argument("--preds-out", help="raw probabilities per record (input to calibrate/thresholds)")
    p.add_argument("--report-out", help="full JSON report incl. confusion and reliability bins")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("synth", help="generate synthetic records with a teacher LLM")
    p.add_argument("--schema", required=True)
    p.add_argument("--recipe", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    p.add_argument("--model")
    p.add_argument("--base-url")
    p.add_argument("--scale", type=float, default=1.0, help="multiply n_per_scenario (0.1 for a dry run)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--no-verify", action="store_true", help="skip the blind re-labelling pass")
    p.set_defaults(fn=cmd_synth)

    p = sub.add_parser("clean", help="dedupe training data and drop rows too close to held-out sets")
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--held-out", nargs="*", help="eval and dev files to protect")
    p.add_argument("--dup", type=float, default=0.9, help="near-duplicate Jaccard threshold")
    p.add_argument("--contam", type=float, default=0.8, help="contamination Jaccard threshold")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_clean)

    for name, fn in (("train-gliner", cmd_train_gliner), ("train-qwen", cmd_train_qwen)):
        p = sub.add_parser(name, help=f"fine-tune ({name.split('-')[1]})")
        p.add_argument("--schema", required=True)
        p.add_argument("--train", nargs="+", required=True)
        p.add_argument("--dev")
        p.add_argument("--out", required=True)
        p.add_argument("--epochs", type=int, default=3)
        p.add_argument("--batch-size", type=int, default=8)
        if name == "train-gliner":
            p.add_argument("--base", default="fastino/GLiNER2.5-Decide")
            p.add_argument("--encoder-lr", type=float, default=1e-5)
            p.add_argument("--task-lr", type=float, default=5e-4)
            p.add_argument("--lora", action="store_true")
            p.add_argument("--augment", type=int, default=2, help="copies with re-shuffled options")
        else:
            p.add_argument("--base", default="Qwen/Qwen3-0.6B")
            p.add_argument("--lr", type=float, default=2e-4)
            p.add_argument("--grad-accum", type=int, default=2)
            p.add_argument("--lora-r", type=int, default=16)
        p.set_defaults(fn=fn)

    p = sub.add_parser("calibrate", help="fit a temperature per question")
    p.add_argument("--schema", required=True)
    p.add_argument("--preds", required=True, help="predictions on a DEV split")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("thresholds", help="pick escalation cut-offs per question")
    p.add_argument("--schema", required=True)
    p.add_argument("--preds", required=True, help="predictions on a DEV split")
    p.add_argument("--calibration")
    p.add_argument("--target", type=float, default=0.97, help="accuracy required on auto-accepted rows")
    p.add_argument("--target-q", nargs="*", help="per-question override, e.g. row_action=0.99")
    p.add_argument("--min-support", type=int, default=20)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_thresholds)

    p = sub.add_parser("serve", help="run the gate as an HTTP service")
    p.add_argument("--schema", required=True)
    _add_backend_args(p)
    p.add_argument("--calibration")
    p.add_argument("--thresholds")
    p.add_argument("--log", default="runs/decisions.jsonl")
    p.add_argument("--corrections", default="data/corrections.jsonl")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8077)
    p.set_defaults(fn=cmd_serve)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
