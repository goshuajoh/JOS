"""Step 1 tooling: static enums, frozen eval set, inter-annotator agreement."""
import json
from pathlib import Path

import pytest

from s1gate.agreement import agreement, cohen_kappa
from s1gate.cli import main
from s1gate.codegen import check_generated, render
from s1gate.frozen import FrozenOverlapError, guard_preds, guard_training, manifest_path
from s1gate.records import Record, read_jsonl, write_jsonl
from s1gate.schema import DecisionSchema

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "configs" / "schema.pos.yaml"
EVAL = ROOT / "data" / "eval" / "eval.sample.jsonl"
GENERATED = ROOT / "s1gate" / "generated" / "pos_row_gate.py"


def test_schema_has_no_op_and_ask_human():
    s = DecisionSchema.load(SCHEMA)
    q = s.question("row_action")
    assert {"no_op", "ask_human"} <= set(q.labels) and q.fallback == "ask_human"
    assert len(q.labels) <= 8  # the guide's limit for one question


def test_generated_enums_match_the_schema():
    s = DecisionSchema.load(SCHEMA)
    assert check_generated(s, GENERATED) is None, "run: s1gate export-enums --schema configs/schema.pos.yaml"
    assert GENERATED.read_text() == render(s, "configs/schema.pos.yaml", "s1gate/generated/pos_row_gate.py")


def test_generated_enums_are_usable_types():
    from s1gate.generated.pos_row_gate import FALLBACKS, PosRowGateDecision, RowAction

    assert RowAction.NO_OP.value == "no_op" and RowAction.ASK_HUMAN.value == "ask_human"
    assert FALLBACKS["row_action"] is RowAction.ASK_HUMAN
    d = PosRowGateDecision(row_action="no_op", needs_human="no", severity="1")
    assert d.row_action is RowAction.NO_OP
    with pytest.raises(Exception):
        PosRowGateDecision(row_action="skip", needs_human="no", severity="1")


def test_check_generated_detects_stale_module(tmp_path):
    s = DecisionSchema.load(SCHEMA)
    stale = tmp_path / "m.py"
    stale.write_text(GENERATED.read_text().replace(s.fingerprint(), "000000000000"))
    assert "fingerprint" in check_generated(s, stale)


def test_freeze_then_guard(tmp_path):
    data = tmp_path / "eval.jsonl"
    data.write_text(EVAL.read_text())
    assert main(["freeze", "--schema", str(SCHEMA), "--data", str(data)]) == 0
    man = manifest_path(data)
    assert json.loads(man.read_text())["n"] == 26
    frozen = read_jsonl(data)
    clean = [Record(id="t1", text="a completely different training row", labels={})]
    guard_training(clean, [man])
    with pytest.raises(FrozenOverlapError):
        guard_training(clean + [Record(id="copy", text=frozen[3].text + "  ", labels={})], [man])
    with pytest.raises(FrozenOverlapError):
        guard_preds([{"id": frozen[0].id, "probs": {}, "gold": {}}], [man])


def test_cli_refuses_to_calibrate_on_frozen_eval(tmp_path):
    data = tmp_path / "eval.jsonl"
    data.write_text(EVAL.read_text())
    main(["freeze", "--schema", str(SCHEMA), "--data", str(data)])
    preds = tmp_path / "preds.jsonl"
    assert main(["eval", "--schema", str(SCHEMA), "--data", str(data), "--backend", "keyword",
                 "--preds-out", str(preds)]) == 0
    code = main(["calibrate", "--schema", str(SCHEMA), "--preds", str(preds), "--out", str(tmp_path / "c.json"),
                 "--frozen", str(manifest_path(data))])
    assert code == 2


def test_eval_warns_when_frozen_file_was_edited(tmp_path, capsys):
    data = tmp_path / "eval.jsonl"
    data.write_text(EVAL.read_text())
    main(["freeze", "--schema", str(SCHEMA), "--data", str(data)])
    rows = read_jsonl(data)
    rows[0].labels["severity"] = "1"
    write_jsonl(data, rows)
    main(["eval", "--schema", str(SCHEMA), "--data", str(data), "--backend", "keyword"])
    assert "changed since it was frozen" in capsys.readouterr().err


def test_agreement_and_kappa(tmp_path):
    s = DecisionSchema.load(SCHEMA)
    a = read_jsonl(EVAL)
    b = [r.model_copy(deep=True) for r in a]
    b[0].labels["row_action"] = "normalize"
    rep = agreement(a, b, s, 0.9)
    ra = rep["questions"]["row_action"]
    assert ra["n"] == 26 and abs(ra["agreement"] - 25 / 26) < 1e-4 and ra["passes"]
    assert ra["top_disagreements"][0] == {"a": "commit", "b": "normalize", "count": 1}
    assert cohen_kappa([("x", "x"), ("y", "y")]) == 1.0
    assert abs(cohen_kappa([("x", "y"), ("y", "x")]) + 1.0) < 1e-9


def test_export_enums_survives_long_backslash_tokens_and_spaces_in_paths(tmp_path):
    import yaml

    from s1gate.codegen import _doc
    token = "\\\\fileserver01\\POS_Uploads\\Distributors\\EMEA\\Northwind_Electronics\\2026\\Uploaded_raw_sheet.xlsx"
    for n in range(40, 140):  # every split position of the long token
        compile("def f():\n" + _doc("x" * n + " " + token + ' "end"') + "\n", "x", "exec")
    data = yaml.safe_load(SCHEMA.read_text())
    data["questions"][0]["prompt"] += " See " + token
    src = tmp_path / "My Schemas" / "schema.yaml"
    src.parent.mkdir()
    src.write_text(yaml.safe_dump(data, allow_unicode=True))
    out = tmp_path / "gen out" / "mod.py"
    assert main(["export-enums", "--schema", str(src), "--out", str(out)]) == 0
    compile(out.read_text(), str(out), "exec")
    assert main(["export-enums", "--schema", str(src), "--out", str(out), "--check"]) == 0
    out.write_text(out.read_text() + "# edited\n")
    assert main(["export-enums", "--schema", str(src), "--out", str(out), "--check"]) == 1


def test_export_enums_refuses_names_that_would_not_import(tmp_path):
    import shlex

    import yaml
    def schema_file(name, options):
        src = tmp_path / f"{name}.yaml"
        src.write_text(yaml.safe_dump({"name": "g", "version": 1,
                                       "questions": [{"name": name, "prompt": "?", "options": options}]}))
        return src

    for name, options, why in (("row_action", {"no-op": "skip", "no_op": "also skip", "commit": "write"}, "both become"),
                               ("class", {"a": "x", "b": "y"}, "keyword"),
                               ("model_config", {"a": "x", "b": "y"}, "pydantic")):
        src = schema_file(name, options)
        out = tmp_path / f"{name}.py"
        assert main(["export-enums", "--schema", str(src), "--out", str(out)]) == 2 and not out.exists()
        if why:
            with pytest.raises(ValueError, match=why):
                render(DecisionSchema.load(src))
    for name in ("match", "case", "type", "model_number", "copy"):  # legal names export and import
        src = schema_file(name, {"a": "x", "b": "y"})
        out = tmp_path / f"ok_{name}.py"
        assert main(["export-enums", "--schema", str(src), "--out", str(out)]) == 0
        assert main(["export-enums", "--schema", str(src), "--out", str(out), "--check"]) == 0
    # the regenerate command in the header is quoted, so it works for paths with spaces
    src = tmp_path / "My Schemas" / "s.yaml"
    src.parent.mkdir()
    src.write_text(SCHEMA.read_text())
    out = tmp_path / "gen out" / "m.py"
    assert main(["export-enums", "--schema", str(src), "--out", str(out)]) == 0
    line = next(x.strip() for x in out.read_text().splitlines() if x.strip().startswith("s1gate export-enums"))
    argv = shlex.split(line)[1:]
    assert argv[argv.index("--schema") + 1] == src.as_posix() and argv[argv.index("--out") + 1] == out.as_posix()


def test_frozen_guard_finds_the_project_eval_from_a_nested_repo_and_reads_old_manifests(tmp_path, monkeypatch):
    import locale

    from s1gate.frozen import load

    proj = tmp_path / "proj"
    (proj / "data" / "eval").mkdir(parents=True)
    (proj / "pyproject.toml").write_text("[project]\nname='x'\n")
    rows = read_jsonl(ROOT / "data" / "eval" / "eval.sample.jsonl")
    write_jsonl(proj / "data" / "eval" / "ev.jsonl", rows)
    assert main(["freeze", "--schema", str(SCHEMA), "--data", str(proj / "data" / "eval" / "ev.jsonl")]) == 0
    nested = proj / "data" / "teacher_repo"  # a git submodule checkout holds a .git file
    nested.mkdir()
    (nested / ".git").write_text("gitdir: ../../.git/modules/teacher_repo\n")
    write_jsonl(nested / "train.jsonl", rows[:5])
    monkeypatch.chdir(tmp_path)  # run from outside the project
    assert main(["clean", "--train", str(nested / "train.jsonl"), "--out", str(tmp_path / "o.jsonl")]) == 2
    # a manifest written in the locale's encoding (cp1252 on Windows) still loads
    man = proj / "data" / "eval" / "ev.frozen.json"
    data = json.loads(man.read_text(encoding="utf-8"))
    data["note"] = "proxy eval — replace with real rows"
    man.write_bytes(json.dumps(data, ensure_ascii=False).encode("cp1252"))
    monkeypatch.setattr(locale, "getpreferredencoding", lambda *_: "cp1252")
    assert load(man)["note"].endswith("real rows")
