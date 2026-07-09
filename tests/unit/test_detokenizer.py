"""The incremental (streaming) detokenizer must reconstruct exactly the same
text as a full decode — token by token, including multi-byte characters."""

import pytest

pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")


@pytest.mark.parametrize(
    "text",
    [
        "Hello, world! This is a plain sentence.",
        "Code: def f(x): return x * 2  # comment",
        "Multi-byte: café, naïve, emoji party time",
        "Numbers 1234567890 and symbols @#$%^&*()",
    ],
)
def test_incremental_equals_full(tok, text):
    from inferneo.tokenizer.tokenizer import IncrementalDetokenizer

    ids = tok(text, add_special_tokens=False).input_ids
    det = IncrementalDetokenizer(tok)
    streamed = ""
    for i in range(1, len(ids) + 1):
        streamed += det.decode(ids[:i])  # feed the growing sequence, one more each step
    full = tok.decode(ids, skip_special_tokens=True)
    assert streamed == full
    assert det.text == full


def test_partial_multibyte_is_held_back(tok):
    """A trailing incomplete UTF-8 char yields no delta until it completes."""
    from inferneo.tokenizer.tokenizer import IncrementalDetokenizer

    ids = tok("emoji 🎉 end", add_special_tokens=False).input_ids
    det = IncrementalDetokenizer(tok)
    for i in range(1, len(ids) + 1):
        delta = det.decode(ids[:i])
        assert "�" not in delta  # never emit the replacement char
    assert det.text == tok.decode(ids, skip_special_tokens=True)
