import importlib.util
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def convert_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(float16="fp16"))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoModelForCausalLM=object(), AutoTokenizer=object()),
    )
    monkeypatch.setitem(
        sys.modules, "model_integrity", types.SimpleNamespace(inventory=lambda _: {})
    )
    script = (
        Path(__file__).parents[1]
        / "models"
        / "optimized-llama"
        / "scripts"
        / "convert_gguf_to_hf.py"
    )
    spec = importlib.util.spec_from_file_location("convert_gguf_to_hf", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_gguf_tokenizer_uses_fast_only_format(tmp_path: Path, convert_module) -> None:

    class Tokenizer:
        def save_pretrained(self, output, **kwargs):
            assert Path(output) == tmp_path
            assert kwargs == {"legacy_format": False}
            (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
            (tmp_path / "tokenizer_config.json").write_text(
                '{"fix_mistral_regex": true, "tokenizer_class": "LlamaTokenizer"}',
                encoding="utf-8",
            )

    convert_module.save_fast_tokenizer(Tokenizer(), tmp_path)
    assert (tmp_path / "tokenizer.json").is_file()
    assert "fix_mistral_regex" not in (
        tmp_path / "tokenizer_config.json"
    ).read_text(encoding="utf-8")


def test_gguf_tokenizer_rejects_copied_model(tmp_path: Path, convert_module) -> None:

    class BrokenTokenizer:
        def save_pretrained(self, output, **kwargs):
            (Path(output) / "tokenizer_config.json").write_text(
                "{}", encoding="utf-8"
            )
            (Path(output) / "tokenizer.model").write_bytes(b"GGUF")

    with pytest.raises(RuntimeError, match="copied from the GGUF"):
        convert_module.save_fast_tokenizer(BrokenTokenizer(), tmp_path)


def test_gguf_tokenizer_loads_named_gguf(convert_module) -> None:
    calls = []

    class Factory:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.append((args, kwargs))
            return object()

    convert_module.AutoTokenizer = Factory
    convert_module.load_gguf_tokenizer("/models/llama")
    assert calls == [
        (
            ("/models/llama",),
            {"gguf_file": "model.gguf"},
        )
    ]
