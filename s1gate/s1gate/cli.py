"""s1gate command line. One subcommand per step; run `s1gate <cmd> -h` for options.

  1   validate, export-schema, export-enums, label, split, freeze, agreement, audit
  2   eval, compare
  3   synth, procgen, blind-export, blind-merge, mine, clean
  4-5 train-gliner, train-qwen
  6   calibrate, report
  7   thresholds
  8   serve, queue, loop-report, build-train, bundle, promote
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
    if args.backend == "frontier":
        kw.update(provider=args.provider, base_url=getattr(args, "base_url", None), guide=args.guide)
    return make_backend(args.backend, model=getattr(args, "model", None),
                        adapter=getattr(args, "adapter", None), url=getattr(args, "url", None),
                        device=getattr(args, "device", None), **kw)


COVERAGE_TARGETS = (0.8, 0.85, 0.9, 0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 0.999)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    k = min(len(xs) - 1, max(0, round(q / 100 * (len(xs) - 1))))
    return xs[k]


def _frozen_manifests(args):
    """--frozen if given, else every data/eval/*.frozen.json in the working directory and in the
    project around each input (the schema, preds, data, training files): the nearest folder
    above the input that has data/eval, never above the project root (pyproject.toml, setup.py), so
    a run from another directory is guarded and an unrelated project is not. A manifest that is
    not readable is skipped with a warning. Says so when nothing is found."""
    from .frozen import DEFAULT_DIR, discover
    from .frozen import load as load_manifest

    explicit = getattr(args, "frozen", None)
    if explicit:
        return discover(explicit)
    roots = [DEFAULT_DIR]
    for attr in ("schema", "preds", "data", "train", "dev", "corrections", "synthetic", "held_out"):
        value = getattr(args, attr, None)
        for item in value if isinstance(value, list) else [value] if isinstance(value, str) else []:
            for parent in list(Path(item).resolve().parents)[:5]:
                if (parent / DEFAULT_DIR).is_dir():
                    roots.append(parent / DEFAULT_DIR)
                    break
                if any((parent / m).exists() for m in ("pyproject.toml", "setup.py", "setup.cfg")):
                    break  # the project root, and it has no frozen eval
    found, seen = [], set()
    for root in roots:
        for m in discover(None, root):
            if m.resolve() in seen:
                continue
            seen.add(m.resolve())
            try:
                load_manifest(m)
            except (OSError, ValueError) as e:
                print(f"warning: skipping unreadable manifest {m}: {e}", file=sys.stderr)
                continue
            found.append(m)
    if not found:
        print(f"warning: no frozen eval manifest found ({DEFAULT_DIR}/*.frozen.json here or near the inputs); "
              "the frozen-eval guard is off for this step - pass --frozen", file=sys.stderr)
    return found


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


def cmd_audit(args) -> int:
    """Check labels against a domain's structural rules (e.g. restricted=yes must be compliance)."""
    from collections import Counter

    from .domains import get as get_domain

    dom = get_domain(args.domain)
    catalogue = dom.load_catalogue(args.catalogue) if args.catalogue else None
    rules: Counter = Counter()
    levels: Counter = Counter()
    out_rows = []
    total = 0
    for path in args.data:
        for r in read_jsonl(path):
            total += 1
            for f in dom.check_labels(r.text, r.labels, catalogue):
                levels[f.level] += 1
                rules[(f.level, f.rule)] += 1
                out_rows.append({"file": path, "id": r.id, "level": f.level, "rule": f.rule,
                                 "message": f.message, "labels": r.labels, "text": r.text})
    shown = [o for o in out_rows if o["level"] == "violation" or args.show_warnings]
    for o in shown[:args.max_show]:
        print(f"{o['level']:<9} {o['id']}: {o['message']}")
    flagged = len({(o["file"], o["id"]) for o in out_rows if o["level"] == "violation"})
    print(f"\n{total} records, {flagged} with violations, "
          f"{levels['violation']} violations, {levels['warning']} warnings")
    for (level, rule), n in sorted(rules.items()):
        print(f"  {level:<9} {rule:<14} {n}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            for o in out_rows:
                fh.write(json.dumps(o, ensure_ascii=False) + "\n")
    return 1 if levels["violation"] or (args.strict and levels["warning"]) else 0


def cmd_export_schema(args) -> int:
    schema = _schema(args)
    _write_json(args.out, schema.json_schema())
    print(f"wrote {args.out}")
    return 0


def cmd_export_enums(args) -> int:
    from .codegen import check_generated, write

    if args.check:
        problem = check_generated(_schema(args), args.out)
        print(problem or f"{args.out} matches {args.schema}")
        return 1 if problem else 0
    try:
        write(args.schema, args.out)
    except ValueError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    print(f"wrote {args.out}")
    return 0


def cmd_freeze(args) -> int:
    from .frozen import freeze, manifest_path

    schema = _schema(args)
    errors = validate_records(read_jsonl(args.data), schema)
    if errors:
        print("\n".join(errors[:20]), file=sys.stderr)
        return 1
    man = freeze(args.data, schema, note=args.note or "")
    print(f"froze {man['n']} records of {args.data} (sha256 {man['sha256'][:12]}) -> {manifest_path(args.data)}")
    for q, counts in man["label_counts"].items():
        print(f"  {q}: " + ", ".join(f"{o}={n}" for o, n in sorted(counts.items())))
    return 0


def cmd_agreement(args) -> int:
    from .agreement import agreement

    schema = _schema(args)
    report = agreement(read_jsonl(args.a), read_jsonl(args.b), schema, args.min)
    print(f"{report['shared_rows']} rows labelled by both; target agreement {args.min:.0%}")
    ok = report["shared_rows"] > 0
    for q, r in report["questions"].items():
        if not r["n"]:
            print(f"  {q:<14} no shared labels - not measured, so not passed")
            ok = False
            continue
        ok &= r["passes"]
        tops = ", ".join(f"{d['a']}->{d['b']} x{d['count']}" for d in r["top_disagreements"])
        print(f"  {q:<14} n={r['n']:<4} agreement={r['agreement']:.3f} kappa={r['kappa']:.3f} "
              f"{'PASS' if r['passes'] else 'FAIL'}" + (f"  [{tops}]" if tops else ""))
    if args.out:
        _write_json(args.out, report)
    return 0 if ok else 1


def cmd_label(args) -> int:
    """Terminal labelling: number keys pick an option, s skips, q saves and quits."""
    schema = _schema(args)
    records = read_jsonl(args.data)
    if args.source:
        records = [r.model_copy(update={"source": args.source}) for r in records]
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
    from .calibrate import apply_calibration, load_biases, load_calibration
    from .metrics import evaluate_question
    from .thresholds import auto_mask, gate_view, load_rules, load_thresholds, pick_threshold

    schema = _schema(args)
    records = read_jsonl(args.data)
    errors = validate_records(records, schema)
    if errors:
        print("\n".join(errors[:20]), file=sys.stderr)
        return 1
    from .frozen import text_hash, verify_file

    manifest, problem = verify_file(args.data)
    if problem:
        print(f"WARNING: {problem}", file=sys.stderr)
    elif manifest:
        print(f"frozen eval set: {manifest['n']} rows frozen on {manifest['frozen']} (hash ok)")
    backend = _backend(args)
    temps = load_calibration(args.calibration, schema)
    biases = load_biases(args.calibration, schema)
    taus = load_thresholds(args.thresholds, schema)
    rules = load_rules(args.thresholds, schema)
    targets_fit = ({q: v.get("target") for q, v in json.loads(Path(args.thresholds).read_text())["thresholds"].items()}
                   if args.thresholds else {})

    preds, latencies = [], []
    bs = args.batch_size or 16
    for i in range(0, len(records), bs):
        chunk = records[i:i + bs]
        t0 = time.perf_counter()
        out = backend.predict_batch([r.text for r in chunk], schema)
        per = (time.perf_counter() - t0) * 1000 / len(chunk)
        for r, probs in zip(chunk, out):
            latencies.append(per)
            preds.append({"id": r.id, "text_hash": text_hash(r.text), "probs": probs, "gold": r.labels,
                          "latency_ms": round(per, 2)})

    if args.preds_out:
        Path(args.preds_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.preds_out, "w", encoding="utf-8") as fh:
            for p in preds:
                fh.write(json.dumps(p) + "\n")

    reports, extras = [], {}
    for q in schema.questions:
        rows = [p for p in preds if q.name in p["gold"]]
        probs = [apply_calibration(p["probs"][q.name], temps.get(q.name, 1.0), biases.get(q.name)) for p in rows]
        golds = [p["gold"][q.name] for p in rows]
        reports.append(evaluate_question(q.name, probs, golds, q.labels, ordinal=q.kind == "score"))
        if not rows:
            continue
        conf, tops, correct = gate_view(probs, golds)
        support = max(1, min(20, len(rows) // 4))
        extras[q.name] = {"coverage_at_in_sample": {  # tau fitted on these same rows: optimistic
            str(t): pick_threshold(conf, correct, t, support)["coverage"] for t in COVERAGE_TARGETS}}
        if q.name in taus:
            auto = auto_mask(conf, tops, taus[q.name], rules.get(q.name))  # exactly the gate's test
            n_auto = int(auto.sum())
            extras[q.name]["at_threshold"] = {
                "tau": taus[q.name], "target": targets_fit.get(q.name), "coverage": round(n_auto / len(rows), 4),
                "accuracy": round(float(correct[auto].mean()), 4) if n_auto else None,
                "auto_errors": int((1 - correct[auto]).sum())}

    p50 = statistics.median(latencies) if latencies else float("nan")
    p95 = _percentile(latencies, 95)
    print(f"\nbackend={backend.name} records={len(records)} calibrated={bool(temps)} "
          f"latency p50={p50:.1f} p95={p95:.1f} ms/record (batch {bs})")
    header = f"{'question':<16}{'n':>5}{'acc':>8}{'macroF1':>9}{'NLL':>8}{'Brier':>8}{'ECE':>8}{'conf':>8}{'MAE':>7}"
    print(header + "\n" + "-" * len(header))
    for rep in reports:
        print(f"{rep.question:<16}{rep.n:>5}{_fmt(rep.accuracy):>8}{_fmt(rep.macro_f1):>9}"
              f"{_fmt(rep.nll):>8}{_fmt(rep.brier):>8}{_fmt(rep.ece):>8}"
              f"{_fmt(rep.mean_confidence):>8}{_fmt(rep.mae, 2):>7}")
    for q, ex in extras.items():
        if "at_threshold" in ex:
            a = ex["at_threshold"]
            print(f"  {q}: tau={a['tau']} automates {a['coverage']:.1%} at accuracy "
                  f"{_fmt(a['accuracy'])} ({a['auto_errors']} auto-accepted errors)")
    if args.report_out:
        from .frozen import file_sha256

        _write_json(args.report_out, {"backend": backend.name, "model": getattr(args, "model", None),
                                      "adapter": getattr(args, "adapter", None), "data": args.data,
                                      "records": len(records), "latency_p50_ms": p50,
                                      "latency_p95_ms": p95, "calibrated": bool(temps),
                                      "schema_fingerprint": schema.fingerprint(),
                                      "data_sha256": file_sha256(args.data),
                                      "thresholds": args.thresholds,
                                      "questions": [{**r.to_dict(), **extras.get(r.question, {})}
                                                    for r in reports]})
        print(f"\nreport -> {args.report_out}")
    return 0


def cmd_compare(args) -> int:
    """Step 2, Gate B: side-by-side baselines; is the best zero-shot model good enough?"""
    reports = [(Path(p).stem, json.loads(Path(p).read_text())) for p in args.reports]
    targets = dict((k, float(v)) for k, v in (t.split("=") for t in (args.target_q or [])))
    qnames = [q["question"] for q in reports[0][1]["questions"]]
    verdict_ok = True
    for qn in qnames:
        tgt = targets.get(qn, args.target)
        print(f"\n{qn}  (coverage at {tgt:.0%} accuracy, tau fitted in-sample)")
        print(f"  {'report':<22}{'backend':<10}{'n':>5}{'acc':>7}{'F1':>7}{'NLL':>7}{'ECE':>7}{'p50ms':>8}{'p95ms':>8}{'cov@t':>7}")
        best = None
        for name, rep in reports:
            q = next((x for x in rep["questions"] if x["question"] == qn), None)
            if not q:
                continue
            stored = {float(k): v for k, v in (q.get("coverage_at_in_sample") or {}).items()}
            at = min((t for t in stored if t >= tgt - 1e-12), default=None)  # nearest stored target, not lower
            cov = stored.get(at) if at is not None else None
            if at is not None and abs(at - tgt) > 1e-12:
                print(f"  (no stored coverage at {tgt}; using the stricter {at})")
            print(f"  {name[:21]:<22}{rep['backend']:<10}{q['n']:>5}{_fmt(q['accuracy']):>7}{_fmt(q['macro_f1']):>7}"
                  f"{_fmt(q['nll']):>7}{_fmt(q['ece']):>7}{_fmt(rep.get('latency_p50_ms'), 1):>8}"
                  f"{_fmt(rep.get('latency_p95_ms'), 1):>8}{_fmt(cov, 2):>7}")
            nll = q["nll"] if q["nll"] is not None and q["nll"] == q["nll"] else float("inf")  # NaN never wins
            if best is None or nll < best[3]:
                best = (name, q, cov, nll)
        if best:
            good = (best[2] or 0) >= args.min_coverage
            verdict_ok &= good
            print(f"  lowest NLL: {best[0]}; coverage at target {_fmt(best[2], 2)} "
                  f"{'>=' if good else '<'} {args.min_coverage:.0%} needed")
    print("\nGate B: " + ("zero-shot is enough - skip training, go to calibrate" if verdict_ok
                         else "fine-tune (steps 3-5): the best zero-shot model misses the coverage target"))
    return 0


# ---------------------------------------------------------------- step 3

def cmd_synth(args) -> int:
    from .synth import Recipe, generate, make_llm

    schema = _schema(args)
    recipe = Recipe.load(args.recipe)
    teachers = [make_llm(args.provider, args.model, args.base_url)]
    for spec in args.teacher or []:  # provider:model[@base_url]
        provider, _, rest = spec.partition(":")
        model, _, base = rest.partition("@")
        teachers.append(make_llm(provider, model or None, base or None))
    rules, catalogue = None, None
    if args.domain:
        from .domains import get as get_domain
        rules = get_domain(args.domain)
        catalogue = rules.load_catalogue(recipe.parts) if recipe.parts else None
    records = generate(schema, recipe, teachers, seed=args.seed, verify=not args.no_verify,
                       scale=args.scale, rules=rules, catalogue=catalogue)
    n = write_jsonl(args.out, records)
    print(f"wrote {n} synthetic records to {args.out}")
    return 0


def cmd_blind_export(args) -> int:
    """Step 3 blind check, half 1: shuffled text-only batches for independent annotators."""
    from .synth import blind_batches

    records: list[Record] = []
    for path in args.data:
        records.extend(read_jsonl(path))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    batches, key = blind_batches(records, args.batch_size, args.seed, anonymize=not args.keep_ids)
    if args.key_out:
        _write_json(args.key_out, key)
    elif not args.keep_ids:
        print("note: no --key-out given; opaque ids cannot be mapped back", file=sys.stderr)
    for i, batch in enumerate(batches):
        with open(out / f"batch_{i:03d}.texts.jsonl", "w", encoding="utf-8") as fh:
            for row in batch:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(records)} rows -> {len(batches)} blind batches in {out} (label each into batch_NNN.labels.jsonl)")
    return 0


def cmd_blind_merge(args) -> int:
    """Step 3 blind check, half 2: keep rows whose blind labels equal their target labels."""
    from .synth import merge_blind

    schema = _schema(args)
    records: list[Record] = []
    for path in args.data:
        records.extend(read_jsonl(path))
    key = json.loads(Path(args.key).read_text()) if args.key else {}
    for k in args.extra_keys or []:
        key.update(json.loads(Path(k).read_text()))
    blind: dict[str, dict[str, str]] = {}
    for path in args.labels:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    rid = key.get(str(row["id"]), str(row["id"]))
                    blind[rid] = {q: str(row[q]) for q in schema.names if q in row}
    kept, rejected, stats = merge_blind(records, blind, schema)
    if args.domain:
        from .domains import get as get_domain
        dom = get_domain(args.domain)
        catalogue = dom.load_catalogue(args.catalogue) if args.catalogue else None
        ok = []
        for r in kept:
            v = [f.message for f in dom.check_labels(r.text, r.labels, catalogue) if f.level == "violation"]
            if v:
                rejected.append(r.model_copy(update={"meta": {**r.meta, "reject": "rule: " + "; ".join(v)}}))
            else:
                ok.append(r)
        stats["rule_rejected"] = len(kept) - len(ok)
        kept = ok
    write_jsonl(args.out, kept)
    if args.rejected_out:
        write_jsonl(args.rejected_out, rejected)
    stats["kept_after_rules"] = len(kept)
    if args.stats_out:
        _write_json(args.stats_out, stats)
    print(json.dumps(stats, indent=1))
    return 0


def cmd_mine(args) -> int:
    """Step 3: turn an eval report's confusions into recipe weights and hard-pair stubs."""
    import yaml

    from .synth import mine_confusions

    report = json.loads(Path(args.report).read_text())
    recipe = yaml.safe_load(Path(args.recipe).read_text(encoding="utf-8"))
    new, found = mine_confusions(report, recipe, args.min_count, args.boost)
    for c in found[:15]:
        print(f"{c['question']}: {c['gold']} -> {c['pred']} x{c['count']} ({c['rate']:.0%})")
    header = (f"# Mined from {args.report} by `s1gate mine`: scenario weights boosted by confusion rate;\n"
              f"# mined_pairs_todo lists confused pairs that need a `change` and full targets.\n")
    Path(args.out).write_text(header + yaml.safe_dump(new, sort_keys=False, allow_unicode=True, width=110),
                              encoding="utf-8")
    print(f"{len(found)} confusions -> {args.out}")
    return 0


def cmd_procgen(args) -> int:
    """Procedural rows with labels known by construction (POS domain)."""
    from .domains import pos_gen

    records = pos_gen.generate(args.pools, args.catalogue, args.recipe, args.regions,
                               n_per_scenario=args.n_per_scenario, pairs_per_type=args.pairs,
                               seed=args.seed)
    n = write_jsonl(args.out, records)
    print(f"wrote {n} procedural records to {args.out}")
    return 0


def cmd_clean(args) -> int:
    from .frozen import guard_training

    records: list[Record] = []
    for path in args.train:
        records.extend(read_jsonl(path))
    kept, dups = dedupe(records, args.dup, label_aware=not args.no_label_aware)
    held: list[Record] = []
    for path in args.held_out or []:
        held.extend(read_jsonl(path))
    kept, leaks = decontaminate(kept, held, args.contam) if held else (kept, [])
    guard_training(kept, _frozen_manifests(args), "cleaned training")
    write_jsonl(args.out, kept)
    if args.dropped_out:
        write_jsonl(args.dropped_out, [r.model_copy(update={"meta": {**r.meta, "dropped": why}})
                                       for why, rs in (("duplicate", dups), ("held_out", leaks)) for r in rs])
    print(f"{len(records)} in, {len(dups)} duplicates or conflicting copies, {len(leaks)} too close to "
          f"held-out, {len(kept)} -> {args.out}")
    if args.stats_out:
        _write_json(args.stats_out, {"in": len(records), "duplicates": len(dups), "held_out": len(leaks),
                                     "out": len(kept), "dup_threshold": args.dup, "contam_threshold": args.contam,
                                     "label_aware": not args.no_label_aware,
                                     "held_out_files": args.held_out or []})
    return 0


# ---------------------------------------------------------------- steps 4-5

def _train_inputs(args):
    from .frozen import guard_training

    schema = _schema(args)
    train = []
    for path in args.train:
        train.extend(read_jsonl(path))
    dev = read_jsonl(args.dev) if args.dev else None
    manifests = _frozen_manifests(args)
    guard_training(train, manifests, "training")
    if dev:
        guard_training(dev, manifests, "dev")
    return schema, train, dev


def cmd_train_gliner(args) -> int:
    from .train_gliner import PreviousRunError, train

    schema, tr, dev = _train_inputs(args)
    try:
        path = train(tr, dev, schema, args.out, base=args.base, epochs=args.epochs,
                     batch_size=args.batch_size, encoder_lr=args.encoder_lr, task_lr=args.task_lr,
                     use_lora=args.lora, augment=args.augment, lora_r=args.lora_r, seed=args.seed,
                     loss=args.loss, overwrite=args.overwrite)
    except PreviousRunError as e:
        print(f"refused: {e} (--overwrite replaces it once the new run has finished)", file=sys.stderr)
        return 2
    print(f"model -> {path}  (evaluate with: s1gate eval --backend gliner --model {path} ...)")
    return 0


def cmd_train_qwen(args) -> int:
    from .train_qwen import train

    schema, tr, dev = _train_inputs(args)
    path = train(tr, dev, schema, args.out, base=args.base, epochs=args.epochs,
                 batch_size=args.batch_size, grad_accum=args.grad_accum, lr=args.lr,
                 lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                 seed=args.seed)
    print(f"adapter -> {path}  (evaluate with: s1gate eval --backend qwen --model {args.base} --adapter {path} ...)")
    return 0


# ---------------------------------------------------------------- steps 6-7

def cmd_calibrate(args) -> int:
    from .calibrate import fit_all

    from .frozen import guard_preds

    schema = _schema(args)
    preds = _read_preds(args.preds)
    guard_preds(preds, _frozen_manifests(args), "calibration")
    result = fit_all(schema, preds, method=args.method)
    _write_json(args.out, result)
    for q, t in result["temperatures"].items():
        print(f"{q:<16} T={t:<8} {result['detail'][q]}")
        if q in result.get("biases", {}):
            print(f"{'':<16} biases {result['biases'][q]}")
    print(f"-> {args.out}")
    return 0


def cmd_thresholds(args) -> int:
    from .calibrate import load_biases, load_calibration
    from .thresholds import fit_all, parse_rules

    schema = _schema(args)
    targets: float | dict[str, float] = args.target
    if args.target_q:
        targets = {q.name: args.target for q in schema.questions}
        for item in args.target_q:
            name, value = item.split("=")
            targets[name] = float(value)
    from .frozen import guard_preds

    preds = _read_preds(args.preds)
    guard_preds(preds, _frozen_manifests(args), "threshold fitting")
    coverage = {k: float(v) for k, v in (c.split("=") for c in (args.coverage_q or []))}
    rules = parse_rules(args.always_escalate, args.option_tau)
    result = fit_all(schema, preds, load_calibration(args.calibration, schema), targets, args.min_support,
                     biases=load_biases(args.calibration, schema), coverage=coverage, rules=rules)
    _write_json(args.out, result)
    for q, info in result["thresholds"].items():
        head = {k: v for k, v in info.items() if k != "tradeoff"}
        print(f"{q:<16} {head}")
        for row in info.get("tradeoff", []):
            print(f"{'':<18}target {row['target']:.2f}: tau {row['tau']:<7} coverage {row['coverage']:<7} "
                  f"accuracy {_fmt(row['accuracy_at_tau'])}")
    if rules:
        print(f"rules: {rules}")
    print(f"-> {args.out}")
    return 0


def cmd_report(args) -> int:
    from .calibrate import load_biases, load_calibration
    from .report import build_report
    from .thresholds import load_rules, load_thresholds

    schema = _schema(args)
    preds = _read_preds(args.preds)
    page = build_report(schema, preds, load_calibration(args.calibration, schema),
                        load_biases(args.calibration, schema), load_thresholds(args.thresholds, schema),
                        title=args.title, source=args.preds, rules=load_rules(args.thresholds, schema))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(page, encoding="utf-8")
    print(f"report -> {args.out}")
    return 0


# ---------------------------------------------------------------- step 8

def cmd_serve(args) -> int:
    import uvicorn

    from .gate import Gate
    from .serve import create_app

    escalator = None
    if args.escalate_to == "frontier":
        from .backends import make_backend
        escalator = make_backend("frontier", model=args.frontier_model, provider=args.frontier_provider,
                                 base_url=args.frontier_base_url, guide=args.guide)
    gate = Gate.from_files(args.schema, _backend(args), args.calibration, args.thresholds, args.log,
                           escalator=escalator, escalator_min_confidence=args.frontier_min_confidence)
    uvicorn.run(create_app(gate, args.corrections), host=args.host, port=args.port)
    return 0


def cmd_queue(args) -> int:
    from .loop import load_corrections, read_log, review_queue

    done = {c.id for c in load_corrections(args.corrections)}
    queue = review_queue(read_log(args.log), only_escalated=not args.all, since=args.since,
                         questions=args.questions, exclude_ids=done)
    n = write_jsonl(args.out, queue)
    print(f"{n} rows to review -> {args.out}  (label with: s1gate label --schema <schema.yaml> "
          f"--data {args.out} --out {args.corrections} --source correction)")
    return 0


def cmd_loop_report(args) -> int:
    from .loop import load_corrections, loop_report, read_log

    rep = loop_report(read_log(args.log), load_corrections(args.corrections), by=args.by)
    for period, qs in rep["periods"].items():
        print(f"\n{period}")
        for q, r in qs.items():
            change = "" if r["coverage_change"] is None else f" ({r['coverage_change']:+.1%})"
            print(f"  {q:<14} rows={r['rows']:<6} coverage={r['coverage']:.1%}{change} by={r['by']} "
                  f"reviewed={r['reviewed']} overturned={r['overturned']}"
                  + (f" (mean conf {r['overturned_mean_confidence']})" if r["overturned_mean_confidence"] else ""))
    if args.out:
        _write_json(args.out, rep)
    return 0


def cmd_build_train(args) -> int:
    from .frozen import guard_training
    from .loop import build_train, load_corrections

    synthetic: list[Record] = []
    for path in args.synthetic or []:
        synthetic.extend(read_jsonl(path))
    rows, stats = build_train(load_corrections(args.corrections), synthetic, args.correction_weight)
    guard_training(rows, _frozen_manifests(args), "training")
    write_jsonl(args.out, rows)
    print(f"{stats} -> {args.out}")
    return 0


def cmd_bundle(args) -> int:
    from .loop import bundle

    man = bundle(args.name, _schema(args), args.backend, args.model, args.calibration, args.thresholds,
                 adapter=args.adapter, report=args.report, notes=args.notes or "")
    out = args.out or f"runs/bundles/{args.name}.json"
    _write_json(out, man)
    print(f"bundle {args.name} -> {out}")
    return 0


def cmd_promote(args) -> int:
    from .loop import promote

    cand = json.loads(Path(args.candidate).read_text())
    cur = json.loads(Path(args.current).read_text()) if args.current else None
    ok, notes = promote(cand, cur, args.nll_slack, args.coverage_slack, args.accuracy_slack)
    print("\n".join(notes))
    print("PROMOTE" if ok else "KEEP CURRENT")
    return 0 if ok else 1


# ---------------------------------------------------------------- parser

def _add_backend_args(p) -> None:
    p.add_argument("--backend", default="gliner", choices=["keyword", "gliner", "qwen", "endpoint", "frontier"])
    p.add_argument("--model", help="HF id or local path (GLiNER checkpoint or Qwen base); frontier: model name")
    p.add_argument("--adapter", help="LoRA adapter dir (qwen backend)")
    p.add_argument("--url", help="base URL (endpoint backend; frontier: OpenAI-compatible base URL)")
    p.add_argument("--device", help="cpu, cuda, cuda:0, mps")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"], help="frontier backend")
    p.add_argument("--guide", default="docs/labeling_guide.md", help="frontier backend: rules sent with each row")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="s1gate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("validate", help="check a schema and label files")
    p.add_argument("--schema", required=True)
    p.add_argument("--data", nargs="*")
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("audit", help="check labels against a domain's structural rules")
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--domain", default="pos")
    p.add_argument("--catalogue", default="configs/pools/parts.yaml",
                   help="part catalogue, so spacing/case variants of a part are recognised")
    p.add_argument("--out", help="write every finding as JSONL")
    p.add_argument("--show-warnings", action="store_true")
    p.add_argument("--max-show", type=int, default=40)
    p.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
    p.set_defaults(fn=cmd_audit)

    p = sub.add_parser("export-schema", help="write the JSON schema of one gate answer")
    p.add_argument("--schema", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_export_schema)

    p = sub.add_parser("export-enums", help="write the questions as static Pydantic enums (Python module)")
    p.add_argument("--schema", required=True)
    p.add_argument("--out", default="s1gate/generated/pos_row_gate.py")
    p.add_argument("--check", action="store_true", help="only check the module matches the schema")
    p.set_defaults(fn=cmd_export_enums)

    p = sub.add_parser("freeze", help="freeze an eval file: hash it so it can be guarded")
    p.add_argument("--schema", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--note", help="free text stored in the manifest (who labelled, how)")
    p.set_defaults(fn=cmd_freeze)

    p = sub.add_parser("agreement", help="inter-annotator agreement on shared rows (Gate A)")
    p.add_argument("--schema", required=True)
    p.add_argument("--a", required=True, help="labels from annotator A")
    p.add_argument("--b", required=True, help="labels from annotator B (same ids)")
    p.add_argument("--min", type=float, default=0.9, help="agreement a question must reach")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_agreement)

    p = sub.add_parser("label", help="label records in the terminal")
    p.add_argument("--schema", required=True)
    p.add_argument("--data", required=True, help="unlabelled or partly labelled JSONL")
    p.add_argument("--out", required=True)
    p.add_argument("--source", choices=["real", "correction", "proxy"],
                   help="mark labelled rows with this source (reviews of the gate: correction)")
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
    p.add_argument("--calibration", help="apply temperatures (and vector biases) before scoring")
    p.add_argument("--thresholds", help="also report coverage and auto-accept accuracy at the fitted taus")
    p.add_argument("--preds-out", help="raw probabilities per record (input to calibrate/thresholds)")
    p.add_argument("--report-out", help="full JSON report incl. confusion and reliability bins")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("compare", help="side-by-side eval reports; Gate B (is zero-shot enough?)")
    p.add_argument("--reports", nargs="+", required=True, help="JSON files from eval --report-out")
    p.add_argument("--target", type=float, default=0.97)
    p.add_argument("--target-q", nargs="*", help="per-question target, e.g. row_action=0.99")
    p.add_argument("--min-coverage", type=float, default=0.3,
                   help="share of rows the best model must automate at target to skip training")
    p.set_defaults(fn=cmd_compare)

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
    p.add_argument("--teacher", nargs="*",
                   help="more teachers, provider:model[@base_url]; calls rotate, each batch is "
                        "blind-checked by the next teacher")
    p.add_argument("--domain", default="pos", help="drop rows that break this domain's rules ('' to skip)")
    p.set_defaults(fn=cmd_synth)

    p = sub.add_parser("blind-export", help="shuffled text-only batches for blind re-labelling")
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch-size", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--key-out", help="where to write opaque id -> record id (keep it away from annotators)")
    p.add_argument("--keep-ids", action="store_true", help="do not replace record ids with opaque ones")
    p.set_defaults(fn=cmd_blind_export)

    p = sub.add_parser("blind-merge", help="keep rows whose blind labels equal their targets")
    p.add_argument("--schema", required=True)
    p.add_argument("--data", nargs="+", required=True, help="rows with target labels")
    p.add_argument("--labels", nargs="+", required=True, help="blind label files: {id, question: option}")
    p.add_argument("--key", help="opaque id -> record id file from blind-export --key-out")
    p.add_argument("--extra-keys", nargs="*", help="more key files (several export rounds)")
    p.add_argument("--domain", help="also drop rows that break the domain's rules (e.g. pos)")
    p.add_argument("--catalogue", default="configs/pools/parts.yaml")
    p.add_argument("--out", required=True)
    p.add_argument("--rejected-out")
    p.add_argument("--stats-out", help="agreement and rejection stats as JSON")
    p.set_defaults(fn=cmd_blind_merge)

    p = sub.add_parser("mine", help="turn an eval report's confusions into recipe weights + pair stubs")
    p.add_argument("--report", required=True, help="eval --report-out JSON (on DEV)")
    p.add_argument("--recipe", default="configs/synth.pos.yaml")
    p.add_argument("--min-count", type=int, default=3)
    p.add_argument("--boost", type=float, default=2.0)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_mine)

    p = sub.add_parser("procgen", help="procedural POS rows with labels known by construction")
    p.add_argument("--pools", default="data/pools/train/companies.json", help="TRAIN-side company pool")
    p.add_argument("--catalogue", default="configs/pools/parts.yaml")
    p.add_argument("--recipe", default="configs/synth.pos.yaml", help="scenario ids, weights, pair ids")
    p.add_argument("--regions", default="configs/pools/regions.yaml", help="currencies per region")
    p.add_argument("--n-per-scenario", type=int, default=100, help="times each scenario's weight")
    p.add_argument("--pairs", type=int, default=50, help="pairs per hard-pair type")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_procgen)

    p = sub.add_parser("clean", help="dedupe training data and drop rows too close to held-out sets")
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--held-out", nargs="*", help="eval and dev files to protect")
    p.add_argument("--dup", type=float, default=0.9, help="near-duplicate Jaccard threshold")
    p.add_argument("--contam", type=float, default=0.8, help="contamination Jaccard threshold")
    p.add_argument("--no-label-aware", action="store_true",
                   help="drop near-duplicates even when their labels differ (kills hard pairs)")
    p.add_argument("--dropped-out", help="write the dropped rows (with the reason) here")
    p.add_argument("--stats-out", help="counts as JSON")
    p.add_argument("--frozen", nargs="*", help="frozen eval manifests to guard (default: data/eval/*.frozen.json)")
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
        p.add_argument("--frozen", nargs="*", help="frozen eval manifests to guard (default: data/eval/*.frozen.json)")
        if name == "train-gliner":
            p.add_argument("--base", default="fastino/GLiNER2.5-Decide")
            p.add_argument("--encoder-lr", type=float, default=1e-5)
            p.add_argument("--task-lr", type=float, default=5e-4)
            p.add_argument("--lora", action="store_true")
            p.add_argument("--lora-r", type=int, default=16)
            p.add_argument("--augment", type=int, default=1,
                           help="extra static copies (options are re-shuffled on every read anyway)")
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--loss", default="softmax", choices=["softmax", "bce"],
                           help="softmax: cross-entropy over each question's options; bce: gliner2's own")
            p.add_argument("--overwrite", action="store_true",
                           help="replace an earlier run in --out once this one has finished (else training refuses)")
        else:
            p.add_argument("--base", default="Qwen/Qwen3-0.6B")
            p.add_argument("--lr", type=float, default=2e-4)
            p.add_argument("--grad-accum", type=int, default=2)
            p.add_argument("--lora-r", type=int, default=16)
            p.add_argument("--lora-alpha", type=int, default=32)
            p.add_argument("--lora-dropout", type=float, default=0.05)
            p.add_argument("--seed", type=int, default=0)
        p.set_defaults(fn=fn)

    p = sub.add_parser("calibrate", help="fit a temperature per question")
    p.add_argument("--schema", required=True)
    p.add_argument("--preds", required=True, help="predictions on a DEV split")
    p.add_argument("--out", required=True)
    p.add_argument("--method", default="temperature", choices=["temperature", "vector"],
                   help="vector = temperature plus one bias per option (can change the top option)")
    p.add_argument("--frozen", nargs="*", help="frozen eval manifests to guard (default: data/eval/*.frozen.json)")
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("report", help="HTML report: reliability diagrams, coverage curves, confusions")
    p.add_argument("--schema", required=True)
    p.add_argument("--preds", required=True)
    p.add_argument("--calibration")
    p.add_argument("--thresholds")
    p.add_argument("--title", default="s1gate report")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("thresholds", help="pick escalation cut-offs per question")
    p.add_argument("--schema", required=True)
    p.add_argument("--preds", required=True, help="predictions on a DEV split")
    p.add_argument("--calibration")
    p.add_argument("--target", type=float, default=0.97, help="accuracy required on auto-accepted rows")
    p.add_argument("--target-q", nargs="*", help="per-question override, e.g. row_action=0.99")
    p.add_argument("--coverage-q", nargs="*", help="pick tau for a coverage instead, e.g. severity=0.8")
    p.add_argument("--always-escalate", nargs="*",
                   help="options a person always sees, e.g. row_action=compliance_review")
    p.add_argument("--option-tau", nargs="*", help="cut-off for one option, e.g. row_action.normalize=0.8")
    p.add_argument("--min-support", type=int, default=20)
    p.add_argument("--out", required=True)
    p.add_argument("--frozen", nargs="*", help="frozen eval manifests to guard (default: data/eval/*.frozen.json)")
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
    p.add_argument("--escalate-to", default="human", choices=["human", "frontier"],
                   help="frontier: ask an approved frontier model about rows under tau first")
    p.add_argument("--frontier-provider", default="openai", choices=["anthropic", "openai"])
    p.add_argument("--frontier-model")
    p.add_argument("--frontier-base-url", help="e.g. an internal Azure OpenAI deployment")
    p.add_argument("--frontier-min-confidence", type=float, default=0.8)
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("queue", help="escalated rows from the decision log -> a labelling queue")
    p.add_argument("--log", default="runs/decisions.jsonl")
    p.add_argument("--corrections", default="data/corrections.jsonl", help="rows already corrected are skipped")
    p.add_argument("--since", help="YYYY-MM-DD")
    p.add_argument("--questions", nargs="*", help="only rows escalated on these questions")
    p.add_argument("--all", action="store_true", help="queue every logged row (shadow mode)")
    p.add_argument("--out", default="data/review_queue.jsonl")
    p.set_defaults(fn=cmd_queue)

    p = sub.add_parser("loop-report", help="coverage, escalations, overturned auto-decisions per period")
    p.add_argument("--log", default="runs/decisions.jsonl")
    p.add_argument("--corrections", default="data/corrections.jsonl")
    p.add_argument("--by", default="week", choices=["day", "week", "month"])
    p.add_argument("--out")
    p.set_defaults(fn=cmd_loop_report)

    p = sub.add_parser("build-train", help="next training set: corrections first, synthetic second")
    p.add_argument("--corrections", default="data/corrections.jsonl")
    p.add_argument("--synthetic", nargs="*", default=["data/synth/train.jsonl"])
    p.add_argument("--correction-weight", type=int, default=3)
    p.add_argument("--frozen", nargs="*", help="frozen eval manifests to guard (default: data/eval/*.frozen.json)")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_build_train)

    p = sub.add_parser("bundle", help="model + calibration + thresholds under one version name")
    p.add_argument("--schema", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--backend", required=True)
    p.add_argument("--model")
    p.add_argument("--adapter")
    p.add_argument("--calibration")
    p.add_argument("--thresholds")
    p.add_argument("--report", help="eval report on the frozen eval set")
    p.add_argument("--notes")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_bundle)

    p = sub.add_parser("promote", help="does a candidate eval report match or beat the current one?")
    p.add_argument("--candidate", required=True, help="eval report (frozen eval, --calibration, --thresholds)")
    p.add_argument("--current", help="the deployed model's report on the same eval file")
    p.add_argument("--nll-slack", type=float, default=0.0)
    p.add_argument("--coverage-slack", type=float, default=0.0)
    p.add_argument("--accuracy-slack", type=float, default=0.0,
                   help="how much auto-accept accuracy may drop when it is still under target")
    p.set_defaults(fn=cmd_promote)
    return ap


def main(argv: list[str] | None = None) -> int:
    from .frozen import FrozenOverlapError

    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except FrozenOverlapError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
