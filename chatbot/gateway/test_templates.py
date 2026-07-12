# gateway/test_templates.py — pure, no services.
from templates import PRESETS, get_preset


def test_every_preset_builds_and_has_stop():
    for name, p in PRESETS.items():
        out = p["build"]("My College", "the context", "the question?")
        assert "My College" in out, name
        assert "the context" in out, name
        assert "the question?" in out, name
        assert isinstance(p["stop"], list) and p["stop"], name
        # every stop token must actually appear as a control token in the template
        assert any(s in out for s in p["stop"]), f"{name}: no stop token in template"


def test_phi3_matches_original_bytes():
    # Regression guard: phi3 must reproduce the pre-refactor hardcoded prompt.
    expected = (
        "<|user|>\nYou are a helpful assistant for ABC.\n"
        "Answer ONLY using the context below.\n"
        'If the answer is not in the context, say "I don\'t have that information."\n\n'
        "Context:\nCTX\n\nQuestion: Q\n<|end|>\n<|assistant|>\n"
    )
    assert PRESETS["phi3"]["build"]("ABC", "CTX", "Q") == expected


def test_unknown_preset_raises():
    try:
        get_preset("does-not-exist")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown preset")


if __name__ == "__main__":
    test_every_preset_builds_and_has_stop()
    test_phi3_matches_original_bytes()
    test_unknown_preset_raises()
    print("✓ template tests passed")
