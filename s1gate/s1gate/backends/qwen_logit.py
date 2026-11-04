"""A decoder LM (Qwen3 0.6B / 1.7B, Qwen3.5, KEV checkpoints) used as a decision model.

Option-logit readout (the SemIf / KEV / JevK5 trick):
  1. Render the state, the question and the options as lettered choices (A, B, C ...).
  2. Run ONE forward pass and read the logits of the letter tokens at the next position.
  3. Softmax over those letters = a probability per option. Nothing is generated.

`OptionScorer` is shared with train_qwen.py, so training and inference see the exact same
prompt and the exact same readout position.
"""
from __future__ import annotations

from ..schema import DecisionSchema, Question
from .base import Backend, Probs

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

SYSTEM = ("You are a decision model. Read the record, then answer the question with the "
          "single letter of the best option.")


def render_user(text: str, q: Question, order: list[str]) -> str:
    lines = ["Record:", text.strip(), "", f"Question: {q.prompt}", "Options:"]
    for letter, option in zip(LETTERS, order):
        desc = q.options[option]
        lines.append(f"{letter}. {option}" + (f": {desc}" if desc else ""))
    return "\n".join(lines)


class OptionScorer:
    """Owns the model + tokenizer; turns (state, question, option order) into letter logits."""

    def __init__(self, model_id: str = DEFAULT_MODEL, adapter: str | None = None,
                 device: str | None = None, dtype: str | None = None, trainable: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = getattr(torch, dtype) if dtype else (
            torch.bfloat16 if self.device.startswith("cuda") else torch.float32)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"  # so position -1 is the answer slot for every row
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
        self.model = model.to(self.device)
        if not trainable:
            self.model.eval()
        self.letter_ids = self._letter_ids()

    def _letter_ids(self) -> list[int]:
        ids = []
        for letter in LETTERS:
            enc = self.tok.encode(letter, add_special_tokens=False)
            if len(enc) != 1:
                raise ValueError(f"tokenizer splits {letter!r}; pick another letter set")
            ids.append(enc[0])
        return ids

    def prompt(self, text: str, q: Question, order: list[str]) -> str:
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": render_user(text, q, order)}]
        if getattr(self.tok, "chat_template", None):
            try:
                # Qwen3: switch thinking off so the next token is the answer letter.
                return self.tok.apply_chat_template(messages, tokenize=False,
                                                    add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return SYSTEM + "\n\n" + messages[1]["content"] + "\n\nAnswer:"

    def letter_logits(self, prompts: list[str], k_max: int, max_length: int = 1024):
        """Logits [batch, k_max] at the answer position. Gradients flow if the model trains."""
        enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True,
                       max_length=max_length).to(self.device)
        out = self.model(**enc)
        last = out.logits[:, -1, :]
        idx = self.torch.tensor(self.letter_ids[:k_max], device=last.device)
        return last.index_select(1, idx).float()


class QwenLogitBackend(Backend):
    name = "qwen"

    def __init__(self, model_id: str = DEFAULT_MODEL, adapter: str | None = None,
                 device: str | None = None, batch_size: int = 8):
        self.scorer = OptionScorer(model_id, adapter=adapter, device=device)
        self.batch_size = batch_size

    def predict(self, text: str, schema: DecisionSchema) -> Probs:
        return self.predict_batch([text], schema)[0]

    def predict_batch(self, texts: list[str], schema: DecisionSchema) -> list[Probs]:
        torch = self.scorer.torch
        results: list[Probs] = [{} for _ in texts]
        for q in schema.questions:
            order = q.labels  # fixed order at inference; training shuffles it
            for start in range(0, len(texts), self.batch_size):
                chunk = texts[start:start + self.batch_size]
                prompts = [self.scorer.prompt(t, q, order) for t in chunk]
                with torch.no_grad():
                    probs = torch.softmax(self.scorer.letter_logits(prompts, len(order)), dim=-1)
                for i, row in enumerate(probs.cpu().tolist()):
                    results[start + i][q.name] = dict(zip(order, row))
        return results
