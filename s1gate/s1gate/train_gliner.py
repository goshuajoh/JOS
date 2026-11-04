"""Step 4-5, route A: fine-tune GLiNER2.5-Decide on your records.

Converts s1gate records into GLiNER2 training rows:
  {"input": text, "output": {"classifications": [
      {"task": q, "labels": [...shuffled...], "true_label": [gold],
       "prompt": "...", "label_descriptions": {...}}]}}

GLiNER2 scores every option of a task in one forward pass and trains with cross-entropy
over them, which is exactly the "train on option scores" objective.

Option order is re-shuffled every time a row is read (`ShuffledOptions`), so every epoch
sees a new order and the model cannot learn "the answer is usually first". `augment` still
writes extra static copies when data is small. The trainer keeps the checkpoint with the
lowest dev loss (`best/`): select on NLL, not accuracy.
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


def reshuffle_output(output: dict, rng: random.Random) -> dict:
    """A copy of one row's output with every task's option list in a fresh random order."""
    out = dict(output)
    tasks = []
    for item in output.get("classifications", []):
        item = dict(item)
        labels = list(item.get("labels", []))
        rng.shuffle(labels)
        item["labels"] = labels
        tasks.append(item)
    out["classifications"] = tasks
    return out


def shuffled_dataset(rows: list[dict], seed: int = 0, validate: bool = True):
    """An ExtractorDataset whose option order changes on every read (so: every epoch).

    The order is seeded from (torch's per-worker seed, row index, how often this copy of the
    dataset has served the row): DataLoader workers get a fresh seed per epoch, and in the main
    process the access count moves on, so either way each epoch draws a new order.
    """
    import torch
    from gliner2.training.trainer import ExtractorDataset

    class ShuffledOptions(ExtractorDataset):
        def __init__(self, data):
            super().__init__(data, shuffle=True, seed=seed, validate=validate)
            self._reads: dict[int, int] = {}

        def __getitem__(self, idx):
            text, output = super().__getitem__(idx)
            n = self._reads.get(idx, 0)
            self._reads[idx] = n + 1
            rng = random.Random(hash((seed, torch.initial_seed(), idx, n)))
            return text, reshuffle_output(output, rng)

    return ShuffledOptions(rows)


def write_rows(rows: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def train(train_records: list[Record], dev_records: list[Record] | None, schema: DecisionSchema,
          out_dir: str, base: str = "fastino/GLiNER2.5-Decide", epochs: int = 3,
          batch_size: int = 8, encoder_lr: float = 1e-5, task_lr: float = 5e-4,
          use_lora: bool = False, augment: int = 1, use_prompts: bool = True,
          lora_r: int = 16, seed: int = 0) -> str:
    """Train and return the path of the checkpoint to evaluate (`best` with dev, else `final`)."""
    from gliner2 import AutoExtractor
    from gliner2.training.trainer import ExtractorTrainer, TrainingConfig

    out = Path(out_dir)
    rows = to_gliner_rows(train_records, schema, use_prompts, augment, seed=seed)
    write_rows(rows, out / "gliner_train.jsonl")  # for inspection; training reads `rows` reshuffled
    dev_path = None
    if dev_records:
        dev_path = out / "gliner_dev.jsonl"
        write_rows(to_gliner_rows(dev_records, schema, use_prompts, 1, seed=seed + 1), dev_path)

    model = AutoExtractor.from_pretrained(base)
    config = TrainingConfig(
        output_dir=str(out),
        num_epochs=epochs,
        batch_size=batch_size,
        encoder_lr=encoder_lr,
        task_lr=task_lr,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=2 * lora_r,
        eval_strategy="epoch" if dev_path else "no",
        save_best=bool(dev_path),
        metric_for_best="eval_loss",  # dev cross-entropy = dev NLL
        greater_is_better=False,
        seed=seed,
    )
    ExtractorTrainer(model, config).train(train_data=shuffled_dataset(rows, seed=seed),
                                          eval_data=str(dev_path) if dev_path else None)
    return str(out / ("best" if dev_path else "final"))
