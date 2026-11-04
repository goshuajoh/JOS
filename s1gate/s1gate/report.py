"""Step 6-7: an HTML report you can read - reliability diagrams, coverage curves, confusions.

    s1gate report --schema configs/schema.pos.yaml --preds runs/dev_preds.jsonl \
                  --calibration runs/calibration.json --thresholds runs/thresholds.json \
                  --out runs/report.html

No plotting dependency: plain SVG inside one HTML file. Per question it shows
  * the reliability diagram before and after calibration (points under the diagonal are
    overconfident),
  * the coverage curve (accuracy of auto-accepted rows vs share automated) with the chosen tau,
  * the confusion matrix (which two options the model mixes up: usually a schema fix),
  * for score questions, predicted vs gold spread (decision models avoid the ends of scales).
"""
from __future__ import annotations

import html
from collections import Counter

import numpy as np

from .calibrate import apply_calibration
from .metrics import coverage_curve, evaluate_question, probs_matrix
from .schema import DecisionSchema

W, H, PAD = 360, 260, 40

CSS = """
:root { --bg:#ffffff; --fg:#1f2328; --muted:#656d76; --grid:#e6e8eb; --raw:#9aa0a6; --cal:#2f6fd6;
        --bad:#c4402f; --card:#f6f8fa; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#0d1117; --fg:#e6edf3; --muted:#8d96a0; --grid:#30363d; --raw:#6e7681; --cal:#58a6ff;
  --bad:#f47067; --card:#161b22; } }
:root[data-theme="dark"] { --bg:#0d1117; --fg:#e6edf3; --muted:#8d96a0; --grid:#30363d;
  --raw:#6e7681; --cal:#58a6ff; --bad:#f47067; --card:#161b22; }
body { background:var(--bg); color:var(--fg); font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;
       margin:0 auto; max-width:1180px; padding:24px 16px; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:18px; margin:28px 0 8px; }
.muted { color:var(--muted); } .row { display:flex; flex-wrap:wrap; gap:16px; }
.card { background:var(--card); border-radius:8px; padding:12px; }
table { border-collapse:collapse; font-variant-numeric:tabular-nums; }
td, th { padding:3px 8px; text-align:right; border-bottom:1px solid var(--grid); }
th:first-child, td:first-child { text-align:left; }
td.hit { font-weight:600; } td.miss { color:var(--bad); }
svg text { fill:var(--muted); font-size:11px; }
svg .axis { stroke:var(--grid); } svg .diag { stroke:var(--muted); stroke-dasharray:4 4; }
svg .raw { stroke:var(--raw); fill:none; stroke-width:2; } svg .rawdot { fill:var(--raw); }
svg .cal { stroke:var(--cal); fill:none; stroke-width:2; } svg .caldot { fill:var(--cal); }
svg .tau { stroke:var(--bad); stroke-dasharray:3 3; } svg .bar { fill:var(--cal); } svg .bar2 { fill:var(--raw); }
"""


def _x(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return PAD + (v - lo) / (hi - lo) * (W - 2 * PAD)


def _y(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return H - PAD - (v - lo) / (hi - lo) * (H - 2 * PAD)


def _axes(xlabel: str, ylabel: str) -> str:
    parts = [f'<line class="axis" x1="{PAD}" y1="{H - PAD}" x2="{W - PAD}" y2="{H - PAD}"/>',
             f'<line class="axis" x1="{PAD}" y1="{PAD}" x2="{PAD}" y2="{H - PAD}"/>']
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<text x="{_x(t):.0f}" y="{H - PAD + 14}" text-anchor="middle">{t:.0%}</text>')
        parts.append(f'<text x="{PAD - 6}" y="{_y(t) + 4:.0f}" text-anchor="end">{t:.0%}</text>')
        parts.append(f'<line class="axis" x1="{PAD}" y1="{_y(t):.1f}" x2="{W - PAD}" y2="{_y(t):.1f}" opacity="0.5"/>')
    parts.append(f'<text x="{W / 2:.0f}" y="{H - 6}" text-anchor="middle">{html.escape(xlabel)}</text>')
    parts.append(f'<text x="12" y="{H / 2:.0f}" text-anchor="middle" transform="rotate(-90 12 {H / 2:.0f})">{html.escape(ylabel)}</text>')
    return "".join(parts)


def _reliability_svg(series: list[tuple[str, list[dict]]], min_count: int = 5) -> str:
    out = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="reliability diagram">',
           _axes("confidence (bin average)", "share correct"),
           f'<line class="diag" x1="{_x(0)}" y1="{_y(0)}" x2="{_x(1)}" y2="{_y(1)}"/>']
    for cls, bins in series:
        pts = [(b["avg_confidence"], b["accuracy"]) for b in bins if b["count"] >= min_count]
        if pts:
            d = " ".join(f"{_x(c):.1f},{_y(a):.1f}" for c, a in pts)
            out.append(f'<polyline class="{cls}" points="{d}"/>')
            out += [f'<circle class="{cls}dot" cx="{_x(c):.1f}" cy="{_y(a):.1f}" r="3"/>' for c, a in pts]
    out.append(f'<text x="{W - PAD}" y="{PAD - 8}" text-anchor="end">grey = raw, blue = calibrated; bins under {min_count} rows hidden</text>')
    return "".join(out) + "</svg>"


def _coverage_svg(curve: list[dict], tau: float | None) -> str:
    out = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="coverage curve">',
           _axes("coverage: share decided without a person", "accuracy of auto-accepted rows")]
    if curve:
        step = max(1, len(curve) // 150)
        pts = curve[::step] + [curve[-1]]
        out.append('<polyline class="cal" points="' + " ".join(f"{_x(p['coverage']):.1f},{_y(p['accuracy']):.1f}" for p in pts) + '"/>')
        if tau is not None and tau <= 1.0:
            kept = [p for p in curve if p["tau"] >= tau]
            if kept:
                p = kept[-1]
                out.append(f'<line class="tau" x1="{_x(p["coverage"]):.1f}" y1="{PAD}" x2="{_x(p["coverage"]):.1f}" y2="{H - PAD}"/>')
                out.append(f'<text x="{_x(p["coverage"]) + 4:.1f}" y="{PAD + 12}">tau {tau:.2f}: {p["coverage"]:.0%} automated, {p["accuracy"]:.1%} right</text>')
    return "".join(out) + "</svg>"


def _spread_svg(labels: list[str], gold: Counter, pred: Counter) -> str:
    n = max(sum(gold.values()), 1)
    out = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="score spread">',
           _axes("", "share of rows")]
    bw = (W - 2 * PAD) / len(labels)
    for i, lab in enumerate(labels):
        x0 = PAD + i * bw
        for j, (cls, cnt) in enumerate((("bar2", gold[lab]), ("bar", pred[lab]))):
            h = cnt / n * (H - 2 * PAD)
            out.append(f'<rect class="{cls}" x="{x0 + 6 + j * (bw / 2 - 6):.1f}" y="{H - PAD - h:.1f}" width="{bw / 2 - 8:.1f}" height="{h:.1f}"/>')
        out.append(f'<text x="{x0 + bw / 2:.1f}" y="{H - PAD + 26}" text-anchor="middle">{html.escape(lab)}</text>')
    out.append(f'<text x="{W - PAD}" y="{PAD - 8}" text-anchor="end">grey = gold, blue = predicted</text>')
    return "".join(out) + "</svg>"


def _confusion_html(conf: dict[str, dict[str, int]]) -> str:
    labels = list(conf)
    rows = ["<table><tr><th>gold \\ predicted</th>" + "".join(f"<th>{html.escape(l)}</th>" for l in labels) + "</tr>"]
    for g in labels:
        cells = []
        for p in labels:
            v = conf[g][p]
            cls = "hit" if g == p and v else ("miss" if v else "")
            cells.append(f'<td class="{cls}">{v or ""}</td>')
        rows.append(f"<tr><td>{html.escape(g)}</td>" + "".join(cells) + "</tr>")
    return "".join(rows) + "</table>"


def _metrics_row(name: str, rep) -> str:
    f = lambda v: "-" if v is None or v != v else f"{v:.3f}"  # noqa: E731
    return (f"<tr><td>{name}</td><td>{rep.n}</td><td>{f(rep.accuracy)}</td><td>{f(rep.macro_f1)}</td>"
            f"<td>{f(rep.nll)}</td><td>{f(rep.brier)}</td><td>{f(rep.ece)}</td><td>{f(rep.mean_confidence)}</td>"
            f"<td>{'-' if rep.mae is None else f'{rep.mae:.2f}'}</td></tr>")


def build_report(schema: DecisionSchema, preds: list[dict], temperatures: dict[str, float] | None = None,
                 biases: dict[str, dict[str, float]] | None = None, taus: dict[str, float] | None = None,
                 title: str = "s1gate report", source: str = "") -> str:
    temperatures, biases, taus = temperatures or {}, biases or {}, taus or {}
    body = [f"<h1>{html.escape(title)}</h1>",
            f'<div class="muted">{html.escape(source)} · schema {schema.name} v{schema.version} '
            f'({schema.fingerprint()}) · {len(preds)} rows</div>']
    for q in schema.questions:
        rows = [p for p in preds if q.name in p.get("gold", {}) and q.name in p["probs"]]
        if not rows:
            continue
        golds = [r["gold"][q.name] for r in rows]
        raw = [r["probs"][q.name] for r in rows]
        cal = [apply_calibration(p, temperatures.get(q.name, 1.0), biases.get(q.name)) for p in raw]
        ordinal = q.kind == "score"
        r0 = evaluate_question(q.name, raw, golds, q.labels, ordinal)
        r1 = evaluate_question(q.name, cal, golds, q.labels, ordinal)
        P = probs_matrix(cal, q.labels)
        g = np.array([q.labels.index(x) for x in golds])
        curve = coverage_curve(P.max(axis=1), (P.argmax(axis=1) == g).astype(float))
        t = temperatures.get(q.name, 1.0)
        body.append(f"<h2>{html.escape(q.name)} <span class='muted'>({q.kind}, T={t:g}"
                    f"{', vector biases' if q.name in biases else ''}"
                    f"{f', tau={taus[q.name]:.3f}' if q.name in taus else ''})</span></h2>")
        body.append("<table><tr><th></th><th>n</th><th>acc</th><th>macroF1</th><th>NLL</th><th>Brier</th>"
                    "<th>ECE</th><th>conf</th><th>MAE</th></tr>" + _metrics_row("raw", r0) +
                    _metrics_row("calibrated", r1) + "</table>")
        panels = [_reliability_svg([("raw", r0.reliability), ("cal", r1.reliability)]),
                  _coverage_svg(curve, taus.get(q.name))]
        if ordinal:
            panels.append(_spread_svg(q.labels, Counter(golds), Counter(q.labels[i] for i in P.argmax(axis=1))))
        body.append('<div class="row">' + "".join(f'<div class="card">{p}</div>' for p in panels) + "</div>")
        body.append(f'<div class="card" style="margin-top:12px;overflow-x:auto">{_confusion_html(r1.confusion)}</div>')
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>{''.join(body)}</body></html>")
