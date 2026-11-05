"""Model code paths without downloading weights: a tiny random Qwen3 and gliner2's own loader.

Skipped when the optional dependencies are not installed (`pip install -e ".[qwen,gliner]"`).
"""
from pathlib import Path

import pytest

from s1gate.records import read_jsonl
from s1gate.schema import DecisionSchema

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "configs" / "schema.pos.yaml"
EVAL = ROOT / "data" / "eval" / "eval.sample.jsonl"


@pytest.fixture(scope="module")
def tiny_qwen(tmp_path_factory):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3Config"):
        pytest.skip("transformers without Qwen3")
    from tokenizers import decoders, models, pre_tokenizers, trainers

    out = tmp_path_factory.mktemp("tiny_qwen")
    corpus = [r.text for r in read_jsonl(EVAL)] + ["Record: Question: Options: A. B. C. D. E. F. G. H."]
    tok = tokenizers.Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=400, special_tokens=["<unk>", "<|endoftext|>", "<|im_start|>", "<|im_end|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>",
                                                eos_token="<|endoftext|>", pad_token="<|endoftext|>")
    fast.chat_template = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
                          "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    cfg = transformers.Qwen3Config(vocab_size=len(fast), hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                   head_dim=8, max_position_embeddings=4096)
    transformers.Qwen3ForCausalLM(cfg).save_pretrained(out)
    fast.save_pretrained(out)
    return str(out)


def test_qwen_option_logit_backend_and_lora_training(tiny_qwen, tmp_path):
    from s1gate.backends.qwen_logit import QwenLogitBackend
    from s1gate.train_qwen import train

    s = DecisionSchema.load(SCHEMA)
    rows = read_jsonl(EVAL)
    probs = QwenLogitBackend(tiny_qwen, device="cpu").predict_batch([r.text for r in rows[:3]], s)
    for p in probs:
        assert set(p) == set(s.names)
        for q in s.questions:
            assert set(p[q.name]) == set(q.labels) and abs(sum(p[q.name].values()) - 1) < 1e-5
    adapter = train(rows, rows[:8], s, str(tmp_path / "run"), base=tiny_qwen, epochs=1, batch_size=8,
                    grad_accum=1, log=lambda *_: None)
    assert adapter.endswith("best") and (Path(adapter) / "adapter_config.json").exists()
    tuned = QwenLogitBackend(tiny_qwen, adapter=adapter, device="cpu").predict(rows[0].text, s)
    assert abs(sum(tuned["row_action"].values()) - 1) < 1e-5


def test_gliner_rows_pass_gliner2_loader_and_reshuffle_every_read():
    pytest.importorskip("gliner2")
    from gliner2.training.data import DataLoader_Factory

    from s1gate.train_gliner import shuffled_dataset, to_gliner_rows

    s = DecisionSchema.load(SCHEMA)
    rows = to_gliner_rows(read_jsonl(EVAL), s)
    assert len(DataLoader_Factory.load(data=rows, max_samples=-1, shuffle=False, seed=0, validate=True)) == 26
    ds = shuffled_dataset(rows, seed=0)
    orders = {tuple(ds[0][1]["classifications"][0]["labels"]) for _ in range(8)}
    assert len(orders) > 1
    gold = {ds[0][1]["classifications"][0]["true_label"][0] for _ in range(4)}
    assert len(gold) == 1


@pytest.fixture(scope="module")
def tiny_gliner(tmp_path_factory):
    pytest.importorskip("torch")
    gl = pytest.importorskip("gliner2")
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    from tokenizers import models, normalizers, pre_tokenizers, trainers
    try:
        from gliner2.configuration import ExtractorConfig
        from gliner2.inference.engine import SpanExtractor
    except ImportError:
        pytest.skip(f"gliner2 {getattr(gl, '__version__', '?')} internals differ")
    out = tmp_path_factory.mktemp("tiny_gliner")
    corpus = [r.text for r in read_jsonl(EVAL)] + [
        "row_action needs_human severity commit normalize vendor_rename pack_qty_issue "
        "compliance_review duplicate no_op ask_human yes no 0 1 2 3 4 label ( ) : ."]
    tok = tokenizers.Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.train_from_iterator(corpus, trainers.WordPieceTrainer(
        vocab_size=800, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]",
                                                cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    enc = transformers.BertConfig(vocab_size=len(fast) + 16, hidden_size=32, num_hidden_layers=1,
                                  num_attention_heads=2, intermediate_size=64, max_position_embeddings=2048)
    cfg = ExtractorConfig(model_name="local-tiny-bert", architecture="span", max_width=8,
                          counting_layer="count_lstm", attn_implementation="eager")
    SpanExtractor(cfg, encoder_config=enc, tokenizer=fast).save_pretrained(str(out))
    return str(out)


@pytest.mark.parametrize("lora", [False, True])
def test_gliner_training_route_a_end_to_end(tiny_gliner, tmp_path, lora):
    import json

    from s1gate.backends.gliner import GlinerBackend
    from s1gate.train_gliner import train

    s = DecisionSchema.load(SCHEMA)
    rows = read_jsonl(EVAL)
    logs = []
    path = train(rows[:16], rows[16:22], s, str(tmp_path / "run"), base=tiny_gliner, epochs=2, batch_size=8,
                 use_lora=lora, lora_r=4, log=logs.append)
    assert "classification loss: softmax cross-entropy over options" in logs
    sel = json.loads((tmp_path / "run" / "selection.json").read_text())
    assert len(sel["dev_nll"]) >= 2 and path.endswith("selected")
    probs = GlinerBackend(path).predict(rows[0].text, s)
    for q in s.questions:
        assert abs(sum(probs[q.name].values()) - 1) < 1e-5


def test_gliner_fixed_schema_sampling():
    pytest.importorskip("gliner2")
    from s1gate.train_gliner import fixed_schema_sampling

    cfg = fixed_schema_sampling()
    assert (cfg.remove_classification_label_prob, cfg.synthetic_label_prob, cfg.include_true_label_prob) == (0.0, 0.0, 1.0)


def test_gliner_parse_keeps_confident_scores_apart():
    import math

    import numpy as np

    from s1gate.backends.gliner import _parse
    sig = lambda x: float(np.float32(1 / (1 + math.exp(-x))))  # noqa: E731  (gliner2 returns float32)
    for logits in ([-20.0, -14.5, -20.0], [15.0, 22.0, -5.0]):
        payload = [{"label": l, "confidence": sig(z)} for l, z in zip("abc", logits)]
        out = _parse(payload, ["a", "b", "c"])
        assert max(out, key=out.get) == "b"


def test_frontier_without_confidence_does_not_clear_the_floor():
    from s1gate.backends.frontier import parse_answer
    s = DecisionSchema.load(SCHEMA)
    out = parse_answer('{"row_action": {"option": "normalize", "confidence": "low"}, '
                       '"needs_human": {"option": "no"}, "severity": "1"}', s)
    assert out["row_action"]["normalize"] == 0.5  # a coin flip: under any escalation floor
    # a yes/no answer keeps its stated option on top, just above a tie
    assert out["needs_human"]["no"] == 0.51 and out["needs_human"]["yes"] < 0.5


def test_gliner_backend_reads_gliner2_softmax_so_confident_outputs_stay_apart(tiny_gliner):
    import torch

    from s1gate.backends.gliner import GlinerBackend
    s = DecisionSchema.load(SCHEMA)
    text = read_jsonl(EVAL)[0].text
    b = GlinerBackend(tiny_gliner)
    assert b.activation == "softmax" and all(t.get("class_act") == "softmax" for t in b.tasks(s).values())
    base = b.predict(text, s)
    last = [m for m in b.model.classifier.modules() if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        last.bias += 25.0  # every option's logit moves together: the softmax must not change
    shifted = b.predict(text, s)
    for q in s.questions:
        assert max(abs(base[q.name][o] - shifted[q.name][o]) for o in q.labels) < 1e-4


def test_gliner_training_refuses_an_old_run_and_replaces_it_only_when_done(tiny_gliner, tmp_path):
    import json

    from s1gate.train_gliner import PreviousRunError, train

    s = DecisionSchema.load(SCHEMA)
    rows = read_jsonl(EVAL)
    out = tmp_path / "run"
    quiet = dict(batch_size=8, log=lambda *_: None)
    train(rows[:8], rows[8:12], s, str(out), base=tiny_gliner, epochs=2, **quiet)
    first = (out / "selected" / "model.safetensors").read_bytes()
    with pytest.raises(PreviousRunError):
        train(rows[:8], rows[8:12], s, str(out), base=tiny_gliner, epochs=1, **quiet)
    with pytest.raises(Exception):  # a retrain that fails (no such base model) leaves the old run in place
        train(rows[:8], rows[8:12], s, str(out), base=str(tmp_path / "missing"), epochs=1, overwrite=True, **quiet)
    assert (out / "selected" / "model.safetensors").read_bytes() == first
    (out / "notes.txt").write_text("mine")
    train(rows[:8], rows[8:12], s, str(out), base=tiny_gliner, epochs=1, overwrite=True, **quiet)
    sel = json.loads((out / "selection.json").read_text())
    assert set(sel["dev_nll"]) <= {"checkpoint-epoch-1", "final"}  # nothing left from the first run
    assert not (out / "checkpoint-epoch-2").exists() and not (out / ".s1gate-staging").exists()
    assert (out / "notes.txt").read_text() == "mine"  # only run folders are replaced


def test_gliner_rows_list_a_task_before_its_prefix(tmp_path):
    pytest.importorskip("gliner2")
    from s1gate.train_gliner import task_order, to_gliner_rows

    s = DecisionSchema.load(SCHEMA)
    q0 = s.questions[0]
    clash = s.model_copy(update={"questions": [q0.model_copy(update={"name": "risk"}),
                                               q0.model_copy(update={"name": "risk_level"})]})
    assert [q.name for q in task_order(clash.questions)] == ["risk_level", "risk"]
    assert [q.name for q in task_order(s.questions)] == s.names  # no clash: schema order
    rec = read_jsonl(EVAL)[0].model_copy(update={"labels": {"risk": "commit", "risk_level": "normalize"}})
    tasks = to_gliner_rows([rec], clash)[0]["output"]["classifications"]
    assert [t["task"] for t in tasks] == ["risk_level", "risk"]
    # gliner2's training loader now hands each task its own gold
    first = {t: next(c for c in tasks if t.startswith(c["task"]))["true_label"] for t in ("risk: x", "risk_level: x")}
    assert first == {"risk: x": ["commit"], "risk_level: x": ["normalize"]}


def test_frontier_confidence_edge_cases():
    from s1gate.backends.frontier import parse_answer
    s = DecisionSchema.load(SCHEMA)
    for conf, expect_floor in (("true", False), ("Infinity", False), ('"high"', False), ("90", True), ("0.2", False),
                               ("150", False), ("-3", False), ("1e300", False), ("1" + "0" * 400, False)):
        raw = '{"row_action": {"option": "commit", "confidence": %s}, "needs_human": {"option": "no", "confidence": %s}}' % (conf, conf)
        p = parse_answer(raw, s)
        for q in ("row_action", "needs_human"):
            top = max(p[q], key=p[q].get)
            assert top == {"row_action": "commit", "needs_human": "no"}[q]  # the stated option stays on top
            assert (p[q][top] >= 0.8) == expect_floor


def test_gliner_training_keeps_user_files_and_refuses_after_an_interrupted_swap(tiny_gliner, tmp_path):
    from s1gate.train_gliner import PREVIOUS, STAGING, PreviousRunError, train

    s = DecisionSchema.load(SCHEMA)
    rows = read_jsonl(EVAL)
    quiet = dict(batch_size=8, epochs=1, log=lambda *_: None)
    out = tmp_path / "run"
    (out / "logs").mkdir(parents=True)
    (out / "logs" / "serve.log").write_text("mine")  # a user's own logs/ is not a training run
    path = train(rows[:8], rows[8:12], s, str(out), base=tiny_gliner, **quiet)
    assert (out / "logs" / "serve.log").read_text() == "mine" and path.endswith("selected")
    assert not (out / STAGING).exists() and not (out / PREVIOUS).exists()
    (out / PREVIOUS).mkdir()  # what an interrupted swap leaves: train() must not guess
    with pytest.raises(PreviousRunError, match="interrupted"):
        train(rows[:8], rows[8:12], s, str(out), base=tiny_gliner, overwrite=True, **quiet)
    (out / PREVIOUS).rmdir()
    (out / STAGING / "selected").mkdir(parents=True)  # a finished run that was never moved in
    with pytest.raises(PreviousRunError, match="never moved"):
        train(rows[:8], rows[8:12], s, str(out), base=tiny_gliner, overwrite=True, **quiet)
