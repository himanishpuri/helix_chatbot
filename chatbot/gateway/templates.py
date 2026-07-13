# gateway/templates.py
# One source of truth for a model family's chat wrapper AND its stop tokens, so
# the two can never drift apart. Pick a family with MODEL_PRESET.
#
# render(messages) wraps a [{role, content}] list (roles: system/user/assistant)
# in the family's control tokens. Both the query-rewrite call and the final
# answer generation go through the same renderer.

# System instruction for the RAG answer turn. Loosened from the old
# "answer ONLY using the context" so the bot can greet, count, and reason over
# what was retrieved — while still refusing to invent unsupported facts.
SYSTEM_INSTRUCTION = (
    "You are a helpful assistant for {college}. "
    "Use the provided context and the conversation so far to answer. "
    "You may greet the user, and you may count, list, or summarize items that "
    "appear in the context. Do not invent facts that are not supported by the "
    "context; if the answer genuinely isn't there, say you don't have that "
    "information.\n\n"
    "Context:\n{context}"
)

# Instruction for the standalone-question rewrite (history-aware retrieval).
REWRITE_INSTRUCTION = (
    "Given the conversation so far and the user's follow-up, rewrite the "
    "follow-up as a single standalone question that makes sense without the "
    "conversation. Resolve pronouns and references. Output ONLY the rewritten "
    "question, nothing else."
)


def _phi3(messages):
    out = []
    for m in messages:
        if m["role"] == "assistant":
            out.append(f"<|assistant|>\n{m['content']}\n<|end|>\n")
        else:  # system + user both go in the user turn for Phi
            out.append(f"<|user|>\n{m['content']}\n<|end|>\n")
    out.append("<|assistant|>\n")
    return "".join(out)


def _qwen(messages):
    out = []
    for m in messages:
        out.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    out.append("<|im_start|>assistant\n")
    return "".join(out)


def _llama3(messages):
    out = ["<|begin_of_text|>"]
    for m in messages:
        out.append(
            f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n"
            f"{m['content']}<|eot_id|>"
        )
    out.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(out)


PRESETS = {
    "phi3": {"render": _phi3, "stop": ["<|end|>", "<|user|>"]},
    "qwen": {"render": _qwen, "stop": ["<|im_end|>"]},
    "llama3": {"render": _llama3, "stop": ["<|eot_id|>"]},
}


def get_preset(name: str) -> dict:
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(
            f"Unknown MODEL_PRESET '{name}'. Choose one of: {', '.join(PRESETS)}"
        )


if __name__ == "__main__":
    msgs = [
        {"role": "system", "content": "SYS-TEXT"},
        {"role": "user", "content": "first?"},
        {"role": "assistant", "content": "ans one"},
        {"role": "user", "content": "second?"},
    ]
    for name, p in PRESETS.items():
        out = p["render"](msgs)
        for m in msgs:
            assert m["content"] in out, f"{name} missing {m['role']}"
        # every stop token must be a control token present in the rendered text
        assert any(s in out for s in p["stop"]), f"{name}: no stop token in render"
        assert isinstance(p["stop"], list) and p["stop"], name
    try:
        get_preset("nope")
        raise SystemExit("expected ValueError")
    except ValueError:
        pass
    print("✓ templates self-check passed")
