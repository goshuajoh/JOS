"""Step 4-5, route B: LoRA fine-tune of a small decoder (Qwen3 0.6B / 1.7B) as an option scorer.

Objective: cross-entropy over the option letters at the answer position, nothing else.
The model is never trained to generate text, only to put mass on the right letter.

Per example and per epoch, option order is re-shuffled, so the gold letter moves around.
That kills position bias and acts as free data augmentation.

GPU strongly recommended (a 0.6B LoRA run on ~5k examples is minutes on one consumer GPU,
hours on CPU).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

from .backends.qwen_logit import DEFAULT_MODEL, OptionScorer
from .records import Record
from .schema import DecisionSchema, Question

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class Example:
    text: str
    question: Question
    gold: str


def build_examples(records: list[Record], schema: DecisionSchema) -> list[Example]:
    out = []
    for r in records:
        for q in schema.questions:
            gold = r.labels.get(q.name)
            if gold is not None:
                out.append(Example(r.text, q, gold))
    return out


def _batch_tensors(scorer: OptionScorer, batch: list[Example], rng: random.Random | None):
    torch = scorer.torch
    prompts, golds, ks = [], [], []
    for ex in batch:
        order = ex.question.labels[:]
        if rng is not None:
            rng.shuffle(order)
        prompts.append(scorer.prompt(ex.text, ex.question, order))
        golds.append(order.index(ex.gold))
        ks.append(len(order))
    k_max = max(ks)
    logits = scorer.letter_logits(prompts, k_max)
    # Questions in one batch can have different option counts: mask the unused letters.
    mask = torch.arange(k_max, device=logits.device)[None, :] >= torch.tensor(ks, device=logits.device)[:, None]
    logits = logits.masked_fill(mask, float("-inf"))
    return logits, torch.tensor(golds, device=logits.device)


def evaluate(scorer: OptionScorer, examples: list[Example], batch_size: int = 16) -> dict:
    torch = scorer.torch
    scorer.model.eval()
    correct, nll, n = 0, 0.0, 0
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            logits, gold = _batch_tensors(scorer, examples[i:i + batch_size], rng=None)
            logp = torch.log_softmax(logits, dim=-1)
            nll += float(-logp[torch.arange(len(gold)), gold].sum())
            correct += int((logits.argmax(-1) == gold).sum())
            n += len(gold)
    scorer.model.train()
    return {"accuracy": correct / max(n, 1), "nll": nll / max(n, 1), "n": n}


def train(train_records: list[Record], dev_records: list[Record] | None, schema: DecisionSchema,
          out_dir: str, base: str = DEFAULT_MODEL, epochs: int = 3, batch_size: int = 8,
          grad_accum: int = 2, lr: float = 2e-4, lora_r: int = 16, lora_alpha: int = 32,
          lora_dropout: float = 0.05, warmup_ratio: float = 0.05, seed: int = 0,
          log=print) -> str:
    """Train a LoRA adapter; returns the adapter dir (use with --adapter)."""
    import torch
    from peft import LoraConfig, get_peft_model

    rng = random.Random(seed)
    torch.manual_seed(seed)
    scorer = OptionScorer(base, trainable=True)
    lora = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                      target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    scorer.model = get_peft_model(scorer.model, lora)
    scorer.model.print_trainable_parameters()
    scorer.model.train()

    train_ex = build_examples(train_records, schema)
    dev_ex = build_examples(dev_records or [], schema)
    steps_per_epoch = math.ceil(len(train_ex) / (batch_size * grad_accum))
    total_steps = max(1, steps_per_epoch * epochs)
    warmup = max(1, int(total_steps * warmup_ratio))

    params = [p for p in scorer.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    def lr_lambda(s: int) -> float:  # linear warmup, then linear decay to zero
        if s < warmup:
            return (s + 1) / warmup
        return max(0.0, (total_steps - s) / max(1, total_steps - warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    loss_fn = torch.nn.CrossEntropyLoss()

    def optimizer_step():
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()

    out = Path(out_dir)
    best = float("inf")
    step = 0
    for epoch in range(epochs):
        rng.shuffle(train_ex)
        running, running_n, micro = 0.0, 0, 0
        for i in range(0, len(train_ex), batch_size):
            logits, gold = _batch_tensors(scorer, train_ex[i:i + batch_size], rng)
            loss = loss_fn(logits, gold) / grad_accum
            loss.backward()
            running += float(loss) * grad_accum
            running_n += 1
            micro += 1
            if micro % grad_accum == 0:
                optimizer_step()
                step += 1
                if step % 20 == 0:
                    log(f"epoch {epoch} step {step}/{total_steps} train loss {running / running_n:.4f}")
                    running, running_n = 0.0, 0
        if micro % grad_accum:  # flush a partial accumulation at the end of the epoch
            optimizer_step()
            step += 1
        if dev_ex:
            metrics = evaluate(scorer, dev_ex)
            log(f"epoch {epoch} dev {metrics}")
            if metrics["nll"] < best:  # select on NLL: it rewards calibration, not just top-1
                best = metrics["nll"]
                scorer.model.save_pretrained(out / "best")
        scorer.model.save_pretrained(out / "last")
    return str(out / ("best" if dev_ex else "last"))
