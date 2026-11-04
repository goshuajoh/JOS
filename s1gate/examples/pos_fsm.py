"""A minimal POS-ingestion FSM around the gate (step 8). Runs end to end on the keyword floor:

    python examples/pos_fsm.py --data data/eval/eval.sample.jsonl --backend keyword --shadow

States after Classify (typed with the generated enums, so a typo is an import error, not a
silent mis-route):

    commit            -> Validate     normalize      -> Normalize
    vendor_rename     -> Rename       duplicate      -> Drop
    no_op             -> Skip         anything a person must see -> Escalate

A person must see the row when the gate escalated it (confidence under tau, or an
always-escalate rule), when needs_human says yes, or in shadow mode (week one: the gate
decides and logs every row, people still review everything, then compare).

With --simulate-review the rows' gold labels play the reviewer, so the corrections file
fills up exactly as it would in production; feed it to `s1gate loop-report` and
`s1gate build-train`.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from enum import Enum
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from s1gate.backends import make_backend  # noqa: E402
from s1gate.gate import Gate, record_correction  # noqa: E402
from s1gate.generated.pos_row_gate import NeedsHuman, RowAction  # noqa: E402
from s1gate.records import read_jsonl  # noqa: E402


class State(str, Enum):
    VALIDATE = "Validate"
    NORMALIZE = "Normalize"
    RENAME = "Rename"
    DROP = "Drop"
    SKIP = "Skip"
    ESCALATE = "Escalate"


ROUTES = {
    RowAction.COMMIT: State.VALIDATE,
    RowAction.NORMALIZE: State.NORMALIZE,
    RowAction.VENDOR_RENAME: State.RENAME,
    RowAction.DUPLICATE: State.DROP,
    RowAction.NO_OP: State.SKIP,
    # pack_qty_issue, compliance_review and ask_human always need a person
    RowAction.PACK_QTY_ISSUE: State.ESCALATE,
    RowAction.COMPLIANCE_REVIEW: State.ESCALATE,
    RowAction.ASK_HUMAN: State.ESCALATE,
}


def classify(gate: Gate, row_id: str, text: str, shadow: bool = False) -> tuple[State, dict]:
    """The FSM's Classify state handler."""
    d = gate.decide(text, record_id=row_id)
    action = RowAction(d["row_action"].option)
    needs_human = NeedsHuman(d["needs_human"].option)
    if shadow or d["row_action"].escalated and d["row_action"].decided_by != "frontier" \
            or needs_human is NeedsHuman.YES:
        return State.ESCALATE, d
    return ROUTES[action], d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data/eval/eval.sample.jsonl"))
    ap.add_argument("--schema", default=str(ROOT / "configs/schema.pos.yaml"))
    ap.add_argument("--backend", default="keyword")
    ap.add_argument("--model")
    ap.add_argument("--adapter")
    ap.add_argument("--calibration")
    ap.add_argument("--thresholds")
    ap.add_argument("--log", default=str(ROOT / "runs/decisions.jsonl"))
    ap.add_argument("--corrections", default=str(ROOT / "data/corrections.jsonl"))
    ap.add_argument("--shadow", action="store_true", help="people review every row; the gate only logs")
    ap.add_argument("--simulate-review", action="store_true", help="gold labels play the reviewer")
    args = ap.parse_args(argv)

    gate = Gate.from_files(args.schema, make_backend(args.backend, model=args.model, adapter=args.adapter),
                           args.calibration, args.thresholds, args.log)
    routes, agree, reviewed = Counter(), Counter(), 0
    for r in read_jsonl(args.data):
        state, d = classify(gate, r.id, r.text, args.shadow)
        routes[state.value] += 1
        if state is State.ESCALATE and args.simulate_review and r.labels:
            reviewed += 1
            record_correction(args.corrections, r.id, r.text, r.labels, reviewer="simulated",
                              meta={"gate_option": d["row_action"].option, "route": state.value})
        if "row_action" in r.labels:
            agree["row_action"] += d["row_action"].model_option == r.labels["row_action"]
    total = sum(routes.values())
    print(f"{total} rows routed: " + ", ".join(f"{k}={v}" for k, v in routes.most_common()))
    if agree:
        print(f"model top option = gold on row_action for {agree['row_action']}/{total} rows")
    if reviewed:
        print(f"{reviewed} reviewer decisions -> {args.corrections}")
    print(f"decision log -> {args.log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
