"""Step 4-5, route A: fine-tune GLiNER2.5-Decide on your records.

Converts s1gate records into GLiNER2 training rows:
  {"input": text, "output": {"classifications": [
      {"task": q, "labels": [...shuffled...], "true_label": [gold],
       "prompt": "...", "label_descriptions": {...}}]}}

Option order is shuffled per row so the model cannot learn "the answer is usually first".
Each epoch sees one fixed shuffle; write several copies with --augment if data is small.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .records import Record
from .schema import DecisionSchema


def to_gliner_rows(records: list[Record], schema: DecisionSchema, use_prompts: bool = True,
                   augment: int = 1, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for _ in range(augment):
        for r in records:
            cls = []
            for q in schema.questions:
                gold = r.labels.get(q.name)
                if gold is None:
                    continue
                labels = q.labels[:]
                rng.shuffle(labels)
                item = {"task": q.name, "labels": labels, "true_label": [gold]}
                if use_prompts:
                    item["prompt"] = q.prompt
                if any(q.options.values()):
                    item["label_descriptions"] = {o: d for o, d in q.options.items() if d}
                cls.append(item)
            if cls:
                rows.append({"input": r.text, "output": {"classifications": cls}})
    return rows


def write_rows(rows: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def train(train_records: list[Record], dev_records: list[Record] | None, schema: DecisionSchema,
          out_dir: str, base: str = "fastino/GLiNER2.5-Decide", epochs: int = 3,
          batch_size: int = 8, encoder_lr: float = 1e-5, task_lr: float = 5e-4,
          use_lora: bool = False, augment: int = 2, use_prompts: bool = True) -> str:
    """Train and return the path of the final checkpoint (load it with --model)."""
    from gliner2 import AutoExtractor
    from gliner2.training.trainer import ExtractorTrainer, TrainingConfig

    out = Path(out_dir)
    train_path = out / "gliner_train.jsonl"
    write_rows(to_gliner_rows(train_records, schema, use_prompts, augment), train_path)
    dev_path = None
    if dev_records:
        dev_path = out / "gliner_dev.jsonl"
        write_rows(to_gliner_rows(dev_records, schema, use_prompts, 1, seed=1), dev_path)

    model = AutoExtractor.from_pretrained(base)
    config = TrainingConfig(
        output_dir=str(out),
        num_epochs=epochs,
        batch_size=batch_size,
        encoder_lr=encoder_lr,
        task_lr=task_lr,
        use_lora=use_lora,
        eval_strategy="epoch" if dev_path else "no",
    )
    ExtractorTrainer(model, config).train(train_data=str(train_path),
                                          eval_data=str(dev_path) if dev_path else None)
    return str(out / "final")
