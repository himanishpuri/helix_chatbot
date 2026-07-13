# gateway/test_templates.py — pure, no services.
from templates import PRESETS, get_preset, SYSTEM_INSTRUCTION, REWRITE_INSTRUCTION

MSGS = [
    {"role": "system", "content": "SYSTEM-TEXT"},
    {"role": "user", "content": "first question?"},
    {"role": "assistant", "content": "the first answer"},
    {"role": "user", "content": "the follow-up?"},
]


def test_every_preset_renders_all_messages_and_has_stop():
    for name, p in PRESETS.items():
        out = p["render"](MSGS)
        for m in MSGS:
            assert m["content"] in out, f"{name}: missing {m['role']} content"
        assert isinstance(p["stop"], list) and p["stop"], name
        # stop token must appear as a control token in the rendered text
        assert any(s in out for s in p["stop"]), f"{name}: no stop token in render"


def test_render_ends_ready_for_assistant():
    # last thing in the prompt should be the assistant opener, so the model
    # continues as the assistant, not echoes a user turn.
    for name, p in PRESETS.items():
        out = p["render"]([{"role": "user", "content": "hi"}])
        # the assistant opener must be the last role marker in the prompt
        assert "assistant" in out[-40:], f"{name}: render doesn't end on assistant turn"


def test_system_and_rewrite_instructions_format():
    s = SYSTEM_INSTRUCTION.format(college="ABC", context="CTX")
    assert "ABC" in s and "CTX" in s
    assert "standalone" in REWRITE_INSTRUCTION.lower()


def test_unknown_preset_raises():
    try:
        get_preset("does-not-exist")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown preset")


if __name__ == "__main__":
    test_every_preset_renders_all_messages_and_has_stop()
    test_render_ends_ready_for_assistant()
    test_system_and_rewrite_instructions_format()
    test_unknown_preset_raises()
    print("✓ template tests passed")
