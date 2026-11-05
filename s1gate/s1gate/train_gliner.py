"""Step 4-5, route A: fine-tune GLiNER2.5-Decide on your records.

Converts s1gate records into GLiNER2 training rows:
  {"input": text, "output": {"classifications": [
      {"task": q, "labels": [...shuffled...], "true_label": [gold],
       "prompt": "...", "label_descriptions": {...}}]}}

What training does, and how it is made to fit a fixed-schema gate:

* Every option of a question is scored in one forward pass (GLiNER2 encodes the text and all
  option labels together).
* Loss. gliner2 2.1.0 trains classifications with a per-option binary cross-entropy
  (one-vs-rest sigmoids). `loss="softmax"` (the default) replaces it, for classification
  tasks only, with cross-entropy over the options of each question - the objective the gate
  is read with. The patch is applied only when gliner2's loss code is the exact version it
  was written against; otherwise gliner2's own loss is kept and a warning says so.
  `loss="bce"` keeps gliner2's loss.
* Sampling. gliner2's default training sampler is built for zero-shot labels: it drops up to
  half the options, renames labels to "label 1..n" half the time, and re-adds a dropped gold
  only half the time. A gate always sees its full, named option set, so here no option is
  dropped or renamed and the gold is always present.
* Option order is re-shuffled every time a row is read (`ShuffledOptions`), so every epoch
  sees a new order and the model cannot learn "the answer is usually first".
* Checkpoints. Every epoch is saved; the one returned (`selected/`) has the lowest dev NLL
  of the probabilities the gliner backend actually produces - select on NLL, not accuracy.
  With LoRA, each adapter checkpoint is merged into the base weights (`*-merged/`) first, so
  the backend can load what is selected. A run trains in `<out>/.s1gate-staging` and replaces
  the folder's previous run only once it has finished, so an old checkpoint can never be
  selected and a failed retrain (no network for the base model, say) leaves the old run in
  place. A folder that holds an earlier run is refused unless `overwrite=True` / `--overwrite`.
* gliner2's trainer finds a classification's gold by the first task whose name is a prefix of
  the task ("risk" before "risk_level" would hand risk's gold to risk_level). Rows therefore
  list a question before any question whose name is a prefix of its own.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import random
import shutil
import warnings
from pathlib import Path

from .records import Record
from .schema import DecisionSchema

# sha256[:16] of gliner2 2.1.0's SpanExtractorModel._compute_sample_loss source
_KNOWN_LOSS_SOURCE = {"f5dab62d38d57d73"}


def task_order(questions: list) -> list:
    """Schema order, except that a question comes before any question whose name prefixes its own.

    gliner2's training loader gives each task the gold of the first task in the row whose name
    is a prefix of it; with this order that is always the task itself. Unchanged when no name
    prefixes another.
    """
    out: list = []
    for q in questions:
        at = next((i for i, p in enumerate(out) if q.name.startswith(p.name)), len(out))
        out.insert(at, q)
    return out


def to_gliner_rows(records: list[Record], schema: DecisionSchema, use_prompts: bool = True,
                   augment: int = 1, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    order = task_order(schema.questions)
    rows = []
    for _ in range(augment):
        for r in records:
            cls = []
            for q in order:
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


def fixed_schema_sampling():
    """gliner2 SamplingConfig for a gate: full named option set, gold always present."""
    from gliner2.processor import SamplingConfig

    return SamplingConfig(remove_classification_prob=0.0, shuffle_classification_labels=True,
                          remove_classification_label_prob=0.0, synthetic_label_prob=0.0,
                          include_true_label_prob=1.0)


def softmax_cls_loss_patch(model) -> bool:
    """Make single-label classification tasks train with cross-entropy over their options.

    Returns True when applied. Only applied when gliner2's loss method is the version this
    replacement was written from, so a gliner2 upgrade can never silently change training.
    The loss sees only the gold vector, not the task's multi_label flag: a task with exactly one
    gold is trained as single-label. Every s1gate question is single-label (`to_gliner_rows`),
    so this holds for gate training; do not apply the patch to data with multi-label tasks.
    Tasks with zero or several golds, or whose gold vector does not match its options, keep
    gliner2's own loss (which rejects the mismatch, as before).
    """
    import torch
    import torch.nn.functional as F

    base = type(model)
    method = None
    for klass in base.__mro__:
        if "_compute_sample_loss" in vars(klass):
            method = vars(klass)["_compute_sample_loss"]
            break
    if method is None:
        return False
    digest = hashlib.sha256(inspect.getsource(method).encode()).hexdigest()[:16]
    if digest not in _KNOWN_LOSS_SOURCE:
        warnings.warn(f"gliner2 loss code changed (sha {digest}); keeping gliner2's per-option BCE loss")
        return False

    def _compute_sample_loss(self, token_embeddings, embs_per_schema, task_types, structure_labels,
                             device, span_scores_per_schema=None):
        cls_loss = torch.tensor(0.0, device=device)
        struct_loss = torch.tensor(0.0, device=device)
        count_loss = torch.tensor(0.0, device=device)
        span_scores_per_schema = span_scores_per_schema or [None] * len(task_types)
        all_counts, all_p_embs = [], []
        for i, task_type in enumerate(task_types):
            if not embs_per_schema[i]:
                continue
            schema_emb = torch.stack(embs_per_schema[i])
            if task_type == "classifications":
                logits = self.classifier(schema_emb[1:]).squeeze(-1)  # skip the [P] token
                labels = torch.tensor(structure_labels[i], dtype=torch.float, device=device)
                if logits.dim() == 1 and labels.shape == logits.shape and int(labels.sum().item()) == 1:
                    # single-label question: cross-entropy over every option's logit
                    gold = labels.argmax().unsqueeze(0)
                    cls_loss = cls_loss + F.cross_entropy(logits.unsqueeze(0), gold, reduction="sum")
                else:  # no gold, several golds, or a size mismatch: gliner2's own loss (and its error)
                    cls_loss = cls_loss + F.binary_cross_entropy_with_logits(logits, labels, reduction="sum")
            else:
                structure = structure_labels[i]
                if structure[0] == 0:
                    continue
                scores = span_scores_per_schema[i]
                if scores is not None:
                    struct_loss = struct_loss + self.compute_struct_loss_from_scores(scores, structure)
                if task_type != "entities":
                    all_counts.append(min(structure[0], 19))
                    all_p_embs.append(schema_emb[0])
        if all_counts and all_p_embs:
            counts = torch.tensor(all_counts, dtype=torch.long, device=device)
            count_loss = F.cross_entropy(self.count_pred(torch.stack(all_p_embs)), counts, reduction="sum")
        return {"classification": cls_loss, "structure": struct_loss, "count": count_loss}

    model._compute_sample_loss = _compute_sample_loss.__get__(model, base)
    return True


def write_rows(rows: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_lora_checkpoint(adapter_dir: Path, base: str) -> Path:
    """gliner2 saves LoRA checkpoints as PEFT adapters only; the gliner backend loads full models.

    Applies the adapter to the base model, merges it into the weights and saves the result in
    `<adapter_dir>-merged`, a normal checkpoint for --model.
    """
    from gliner2 import AutoExtractor
    from peft import PeftModel

    if (adapter_dir / "config.json").exists():
        return adapter_dir  # already a full model
    merged_dir = adapter_dir.with_name(adapter_dir.name + "-merged")
    model = AutoExtractor.from_pretrained(base)
    merged = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    merged.save_pretrained(str(merged_dir))
    return merged_dir


def select_on_dev_nll(candidates: list[Path], dev_records: list[Record], schema: DecisionSchema,
                      log=print) -> tuple[Path, dict]:
    """Score each checkpoint with the gliner backend on dev; lowest mean NLL over questions wins."""
    from .backends.gliner import GlinerBackend
    from .metrics import evaluate_question

    scores = {}
    for path in candidates:
        backend = GlinerBackend(str(path))
        preds = backend.predict_batch([r.text for r in dev_records], schema)
        nlls = []
        for q in schema.questions:
            rows = [(p[q.name], r.labels[q.name]) for p, r in zip(preds, dev_records) if q.name in r.labels]
            if rows:
                nlls.append(evaluate_question(q.name, [p for p, _ in rows], [g for _, g in rows], q.labels).nll)
        scores[str(path)] = sum(nlls) / len(nlls) if nlls else float("inf")
        log(f"dev NLL {scores[str(path)]:.4f}  {path.name}")
    best = min(candidates, key=lambda p: scores[str(p)])
    return best, scores


STAGING, PREVIOUS = ".s1gate-staging", ".s1gate-previous"
_RUN = {"final", "best", "selected", "selection.json"}                     # only a training run writes these
_RUN_EXTRAS = {"training_config.json", "gliner_train.jsonl", "gliner_dev.jsonl"}  # replaced with it


class PreviousRunError(FileExistsError):
    """The output folder already holds a training run (or the remains of an interrupted one)."""


def run_artifacts(out: Path) -> list[Path]:
    """The training run in `out`: checkpoints, merged copies, best/, final/, selected/, selection.json."""
    if not out.is_dir():
        return []
    return sorted(p for p in out.iterdir()
                  if p.name in _RUN or p.name.startswith("checkpoint-") or p.name.endswith("-merged"))


def _run_extras(out: Path) -> list[Path]:
    """Files a run writes besides the run itself (its config, its rows, gliner2's length caches)."""
    if not out.is_dir():
        return []
    return sorted(p for p in out.iterdir() if p.name in _RUN_EXTRAS or p.name.startswith("gliner_dev.jsonl."))


def _swap_in(stage: Path, out: Path, log=print) -> None:
    """Replace the run in `out` by the finished one in `stage`.

    The old run is first moved aside into `<out>/.s1gate-previous` and deleted only after the new
    one is in place, so an interruption leaves one complete run on disk (train() then refuses to
    start until it is sorted out). gliner2's logs/ is merged into a logs/ folder of the user's, never
    replaces it; any other name that is not the run's own is left to the user.
    """
    previous = out / PREVIOUS
    old = run_artifacts(out) + _run_extras(out)
    if old:
        previous.mkdir()
        for p in old:
            p.rename(previous / p.name)
    for p in sorted(stage.iterdir()):
        target = out / p.name
        if p.name == "logs" and p.is_dir():
            if any(p.iterdir()):
                target.mkdir(exist_ok=True)
                for q in p.iterdir():
                    _remove(target / q.name)
                    q.rename(target / q.name)
            _remove(p)
        elif target.exists() or target.is_symlink():
            log(f"kept your {target}; the run's own {p.name} is dropped")
            _remove(p)
        else:
            p.rename(target)
    stage.rmdir()
    _remove(previous)


def _remove(p: Path) -> None:
    if p.is_dir() and not p.is_symlink():
        shutil.rmtree(p)
    elif p.exists() or p.is_symlink():
        p.unlink()


def train(train_records: list[Record], dev_records: list[Record] | None, schema: DecisionSchema,
          out_dir: str, base: str = "fastino/GLiNER2.5-Decide", epochs: int = 3,
          batch_size: int = 8, encoder_lr: float = 1e-5, task_lr: float = 5e-4,
          use_lora: bool = False, augment: int = 1, use_prompts: bool = True,
          lora_r: int = 16, seed: int = 0, loss: str = "softmax", overwrite: bool = False,
          log=print) -> str:
    """Train and return the checkpoint to evaluate: `selected` (lowest dev NLL) with dev, else `final`."""
    from gliner2 import AutoExtractor
    from gliner2.training.trainer import ExtractorTrainer, TrainingConfig

    if loss not in ("softmax", "bce"):
        raise ValueError("loss must be 'softmax' or 'bce'")
    out = Path(out_dir)
    if (out / PREVIOUS).exists():
        raise PreviousRunError(f"{out / PREVIOUS} holds the run an interrupted swap moved aside; move what "
                               f"you want to keep back into {out}, then delete {PREVIOUS} (and {STAGING})")
    if any((out / STAGING / n).exists() for n in ("selected", "selection.json", "final")):
        raise PreviousRunError(f"{out / STAGING} holds a finished run that was never moved into {out}; "
                               "move it in or delete it first")
    old = run_artifacts(out)
    if old and not overwrite:
        raise PreviousRunError(f"{out} already holds a training run ({', '.join(p.name for p in old[:6])}"
                               f"{', ...' if len(old) > 6 else ''}); use a fresh output folder or overwrite")
    rows = to_gliner_rows(train_records, schema, use_prompts, augment, seed=seed)
    dev_rows = to_gliner_rows(dev_records, schema, use_prompts, 1, seed=seed + 1) if dev_records else None
    model = AutoExtractor.from_pretrained(base)  # fails before anything in `out` is touched

    stage = out / STAGING  # the run is built here and replaces the old one only when complete
    _remove(stage)         # an unfinished earlier attempt (a finished one is refused above)
    stage.mkdir(parents=True)
    write_rows(rows, stage / "gliner_train.jsonl")  # for inspection; training reads `rows` reshuffled
    dev_path = None
    if dev_rows:
        dev_path = stage / "gliner_dev.jsonl"
        write_rows(dev_rows, dev_path)

    model.processor.sampling_config = fixed_schema_sampling()
    patched = softmax_cls_loss_patch(model) if loss == "softmax" else False
    log(f"classification loss: {'softmax cross-entropy over options' if patched else 'gliner2 per-option BCE'}")
    config = TrainingConfig(
        output_dir=str(stage),
        num_epochs=epochs,
        batch_size=batch_size,
        encoder_lr=encoder_lr,
        task_lr=task_lr,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=2 * lora_r,
        save_adapter_only=False,      # LoRA checkpoints are merged into full models after training
        eval_strategy="epoch" if dev_path else "no",
        save_best=bool(dev_path),
        metric_for_best="eval_loss",
        greater_is_better=False,
        save_total_limit=epochs + 1,  # keep every epoch for the NLL selection below
        seed=seed,
    )
    ExtractorTrainer(model, config).train(train_data=shuffled_dataset(rows, seed=seed),
                                          eval_data=str(dev_path) if dev_path else None)
    candidates = sorted(p for p in stage.iterdir() if p.is_dir() and
                        (p.name.startswith("checkpoint-epoch-") or p.name == "final"))
    if use_lora:
        candidates = [merge_lora_checkpoint(c, base) for c in candidates]
    if dev_records:
        best, scores = select_on_dev_nll(candidates, dev_records, schema, log)
        shutil.copytree(best, stage / "selected")
        (stage / "selection.json").write_text(json.dumps(
            {"selected": best.name, "dev_nll": {Path(k).name: v for k, v in scores.items()}}, indent=1))
        result = "selected"
    else:
        result = next(c for c in candidates if c.name.startswith("final")).name

    _swap_in(stage, out, log)
    return str(out / result)
