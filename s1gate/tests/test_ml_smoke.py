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
