"""Step 1, Gate A: inter-annotator agreement on the same rows.

Have a second person label ~50 of the rows the first person labelled. Under about 90%
agreement on a question means its option descriptions are unclear: fix the schema (and the
labelling guide) before labelling more. The most frequent disagreement pairs say where.
"""
from __future__ import annotations

from collections import Counter

from .records import Record
from .schema import DecisionSchema


def cohen_kappa(pairs: list[tuple[str, str]]) -> float:
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def agreement(a: list[Record], b: list[Record], schema: DecisionSchema,
              min_agreement: float = 0.9) -> dict:
    by_id = {r.id: r for r in b}
    out = {}
    for q in schema.questions:
        pairs = [(r.labels[q.name], by_id[r.id].labels[q.name]) for r in a
                 if r.id in by_id and q.name in r.labels and q.name in by_id[r.id].labels]
        n = len(pairs)
        rate = sum(x == y for x, y in pairs) / n if n else float("nan")
        confusions = Counter((x, y) for x, y in pairs if x != y).most_common(5)
        out[q.name] = {
            "n": n,
            "agreement": round(rate, 4) if n else None,
            "kappa": round(cohen_kappa(pairs), 4) if n else None,
            "passes": bool(n) and rate >= min_agreement,
            "top_disagreements": [{"a": x, "b": y, "count": c} for (x, y), c in confusions],
        }
    return {"min_agreement": min_agreement, "shared_rows": len(set(by_id) & {r.id for r in a}),
            "questions": out}
