# gateway/templates.py
# One source of truth for a model family's chat template AND its stop tokens,
# so the two can never drift apart. Pick a family with MODEL_PRESET.
#
# The instruction body is identical across presets — only the wrapping control
# tokens (and matching stop sequences) differ per model family.

_INSTRUCTION = (
    "You are a helpful assistant for {college}.\n"
    "Answer ONLY using the context below.\n"
    'If the answer is not in the context, say "I don\'t have that information."\n\n'
    "Context:\n{context}\n\n"
    "Question: {query}"
)


def _phi3(college, context, query):
    body = _INSTRUCTION.format(college=college, context=context, query=query)
    return f"<|user|>\n{body}\n<|end|>\n<|assistant|>\n"


def _qwen(college, context, query):
    # ChatML — Qwen2.5, and most chatml-tuned models.
    body = _INSTRUCTION.format(college=college, context=context, query=query)
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{body}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _llama3(college, context, query):
    body = _INSTRUCTION.format(college=college, context=context, query=query)
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{body}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


PRESETS = {
    "phi3": {"build": _phi3, "stop": ["<|end|>", "<|user|>"]},
    "qwen": {"build": _qwen, "stop": ["<|im_end|>"]},
    "llama3": {"build": _llama3, "stop": ["<|eot_id|>"]},
}


def get_preset(name: str) -> dict:
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(
            f"Unknown MODEL_PRESET '{name}'. Choose one of: {', '.join(PRESETS)}"
        )


if __name__ == "__main__":
    for name, p in PRESETS.items():
        out = p["build"]("Test College", "some context", "a question?")
        assert "Test College" in out and "some context" in out and "a question?" in out
        assert isinstance(p["stop"], list) and p["stop"], name
    # phi3 must reproduce the original hardcoded bytes exactly (regression guard).
    expected = (
        "<|user|>\nYou are a helpful assistant for ABC.\n"
        "Answer ONLY using the context below.\n"
        'If the answer is not in the context, say "I don\'t have that information."\n\n'
        "Context:\nCTX\n\nQuestion: Q\n<|end|>\n<|assistant|>\n"
    )
    assert PRESETS["phi3"]["build"]("ABC", "CTX", "Q") == expected, "phi3 drift!"
    try:
        get_preset("nope")
        raise SystemExit("expected ValueError")
    except ValueError:
        pass
    print("✓ templates self-check passed")
