"""Unit tests for IndexTTS2ScriptDubbing helper functions.

Tests pure logic only — no model loading required.
Run: python test_script_dubbing.py
"""

import sys
import os
import types

sys.path.insert(0, os.path.dirname(__file__))

# Mock ComfyUI-specific module before importing nodes
folder_paths_mock = types.ModuleType("folder_paths")
folder_paths_mock.models_dir = "/tmp"
sys.modules["folder_paths"] = folder_paths_mock

import numpy as np
import torch

from nodes import _parse_srt, _split_character_dialogue, _extract_emotion_segment, _assemble_timeline


def test_parse_srt_standard():
    """Test standard multi-line SRT format."""
    srt = """1
00:00:01,000 --> 00:00:03,000
唐僧：悟空，你又调皮了。

2
00:00:04,000 --> 00:00:06,500
孙悟空：师父，俺老孙冤枉啊！

3
00:00:07,000 --> 00:00:09,000
这是一句没有角色名的旁白。
"""
    entries = _parse_srt(srt)
    assert len(entries) == 3, f"Expected 3 entries, got {len(entries)}"

    assert entries[0]["index"] == 1
    assert entries[0]["start_ms"] == 1000
    assert entries[0]["end_ms"] == 3000
    assert entries[0]["character"] == "唐僧"
    assert entries[0]["emotion"] == ""
    assert entries[0]["dialogue"] == "悟空，你又调皮了。"

    assert entries[1]["index"] == 2
    assert entries[1]["start_ms"] == 4000
    assert entries[1]["end_ms"] == 6500
    assert entries[1]["character"] == "孙悟空"
    assert entries[1]["dialogue"] == "师父，俺老孙冤枉啊！"

    assert entries[2]["index"] == 3
    assert entries[2]["character"] == ""
    assert entries[2]["dialogue"] == "这是一句没有角色名的旁白。"

    print("  PASS: test_parse_srt_standard")


def test_parse_srt_english_colon():
    """Test SRT with English colon."""
    srt = """1
00:00:00,500 --> 00:00:02,000
Alice: Hello world!
"""
    entries = _parse_srt(srt)
    assert len(entries) == 1
    assert entries[0]["character"] == "Alice"
    assert entries[0]["emotion"] == ""
    assert entries[0]["dialogue"] == "Hello world!"
    print("  PASS: test_parse_srt_english_colon")


def test_parse_srt_with_emotion():
    """Test SRT with emotion descriptions in parentheses."""
    srt = """1
00:00:01,000 --> 00:00:03,000
唐僧(高兴的说)：悟空，快来看。

2
00:00:04,000 --> 00:00:06,000
孙悟空（愤怒）：俺老孙不服！

3
00:00:07,000 --> 00:00:09,000
唐僧：普通台词，无情绪标注。
"""
    entries = _parse_srt(srt)
    assert len(entries) == 3

    assert entries[0]["character"] == "唐僧"
    assert entries[0]["emotion"] == "高兴的说"
    assert entries[0]["dialogue"] == "悟空，快来看。"

    assert entries[1]["character"] == "孙悟空"
    assert entries[1]["emotion"] == "愤怒"
    assert entries[1]["dialogue"] == "俺老孙不服！"

    assert entries[2]["character"] == "唐僧"
    assert entries[2]["emotion"] == ""
    assert entries[2]["dialogue"] == "普通台词，无情绪标注。"

    print("  PASS: test_parse_srt_with_emotion")


def test_parse_srt_empty():
    """Test empty / invalid SRT."""
    assert _parse_srt("") == []
    assert _parse_srt("just some text") == []
    assert _parse_srt("\n\n\n") == []
    print("  PASS: test_parse_srt_empty")


def test_parse_srt_timestamps():
    """Test precise timestamp parsing."""
    srt = """1
01:23:45,678 --> 02:00:00,000
Test
"""
    entries = _parse_srt(srt)
    assert len(entries) == 1
    # 1*3600000 + 23*60000 + 45*1000 + 678 = 5025678
    assert entries[0]["start_ms"] == 5025678
    # 2*3600000 = 7200000
    assert entries[0]["end_ms"] == 7200000
    print("  PASS: test_parse_srt_timestamps")


def test_split_character_dialogue():
    """Test character/emotion/dialogue splitting."""
    # Chinese colon, no emotion
    c, e, d = _split_character_dialogue("唐僧：悟空，快走。")
    assert c == "唐僧" and e == "" and d == "悟空，快走。"

    # English colon, no emotion
    c, e, d = _split_character_dialogue("Alice: Hello!")
    assert c == "Alice" and e == "" and d == "Hello!"

    # No character
    c, e, d = _split_character_dialogue("这是一段旁白")
    assert c == "" and e == "" and d == "这是一段旁白"

    # Colon in dialogue
    c, e, d = _split_character_dialogue("角色: 说话：更多内容")
    assert c == "角色" and d == "说话：更多内容"

    # With emotion — half-width parentheses
    c, e, d = _split_character_dialogue("唐僧(高兴的说): 悟空，快走。")
    assert c == "唐僧" and e == "高兴的说" and d == "悟空，快走。"

    # With emotion — full-width parentheses + Chinese colon
    c, e, d = _split_character_dialogue("孙悟空（愤怒）：俺老孙不服！")
    assert c == "孙悟空" and e == "愤怒" and d == "俺老孙不服！"

    # With emotion — mixed
    c, e, d = _split_character_dialogue("唐僧(悲伤地)：阿弥陀佛。")
    assert c == "唐僧" and e == "悲伤地" and d == "阿弥陀佛。"

    print("  PASS: test_split_character_dialogue")


def test_extract_emotion_segment():
    """Test emotion audio segment extraction."""
    sr = 16000
    duration_s = 10.0
    n_samples = int(sr * duration_s)
    # Create a simple ramp signal
    wav = torch.linspace(0, 1, n_samples).unsqueeze(0).unsqueeze(0)  # (1, 1, N)
    audio = {"waveform": wav, "sample_rate": sr}

    # Normal segment: 2s to 5s
    seg, info = _extract_emotion_segment(audio, 2000, 5000)
    expected_len = int(3.0 * sr)
    actual_len = seg["waveform"].shape[-1]
    assert abs(actual_len - expected_len) <= 1, f"Expected ~{expected_len}, got {actual_len}"
    assert seg["sample_rate"] == sr
    assert "切片" in info

    # Too short segment (< 0.1s) -> fallback to full
    seg, info = _extract_emotion_segment(audio, 1000, 1050)
    assert seg["waveform"].shape[-1] == n_samples, "Should fallback to full audio"
    assert "fallback" in info

    # Out of range -> fallback
    seg, info = _extract_emotion_segment(audio, 20000, 25000)
    assert seg["waveform"].shape[-1] == n_samples, "Should fallback to full audio"
    assert "fallback" in info

    print("  PASS: test_extract_emotion_segment")


def test_assemble_timeline():
    """Test timeline assembly."""
    sr = 16000

    entries = [
        {"index": 1, "start_ms": 0, "end_ms": 2000, "character": "A", "dialogue": "Hi"},
        {"index": 2, "start_ms": 3000, "end_ms": 5000, "character": "B", "dialogue": "Hey"},
    ]

    # Create short audio clips (0.5s each)
    clip_len = int(0.5 * sr)
    clip1 = np.ones((1, clip_len), dtype=np.float32) * 0.5
    clip2 = np.ones((1, clip_len), dtype=np.float32) * -0.5

    segments = [
        (entries[0], clip1),
        (entries[1], clip2),
    ]

    result = _assemble_timeline(segments, entries, sr)

    # Check shape: should be (1, N)
    assert result.ndim == 2
    assert result.shape[0] == 1

    # Check clip1 is at position 0
    assert abs(result[0, 0] - 0.5) < 1e-5, f"Expected 0.5 at position 0, got {result[0, 0]}"

    # Check clip2 is at position 3s
    pos_3s = int(3.0 * sr)
    assert abs(result[0, pos_3s] - (-0.5)) < 1e-5, f"Expected -0.5 at position 3s, got {result[0, pos_3s]}"

    # Check silence between clips (at 1s, should be ~0)
    pos_1s = int(1.0 * sr)
    assert abs(result[0, pos_1s]) < 1e-5, f"Expected silence at 1s, got {result[0, pos_1s]}"

    print("  PASS: test_assemble_timeline")


def test_assemble_timeline_empty():
    """Test that empty segments raise RuntimeError."""
    try:
        _assemble_timeline([], [{"end_ms": 1000}], 16000)
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass
    print("  PASS: test_assemble_timeline_empty")


def test_assemble_timeline_no_overlap():
    """Test that a long segment pushes the next one forward instead of overlapping."""
    sr = 16000

    entries = [
        {"index": 1, "start_ms": 0, "end_ms": 2000, "character": "A", "dialogue": "Hi"},
        {"index": 2, "start_ms": 1000, "end_ms": 3000, "character": "B", "dialogue": "Hey"},
    ]

    # clip1 is 3s — longer than the 1s gap before clip2's SRT start
    clip1_len = int(3.0 * sr)
    clip2_len = int(1.0 * sr)
    clip1 = np.ones((1, clip1_len), dtype=np.float32) * 0.5
    clip2 = np.ones((1, clip2_len), dtype=np.float32) * -0.5

    segments = [
        (entries[0], clip1),
        (entries[1], clip2),
    ]

    result = _assemble_timeline(segments, entries, sr)

    # clip1 placed at 0s, full 3s preserved
    assert abs(result[0, 0] - 0.5) < 1e-5
    assert abs(result[0, clip1_len - 1] - 0.5) < 1e-5, "clip1 should not be truncated"

    # clip2 pushed to after clip1 ends (at 3s, not 1s)
    assert abs(result[0, clip1_len] - (-0.5)) < 1e-5, (
        f"clip2 should start right after clip1 at sample {clip1_len}"
    )

    # After clip2 ends: silence
    after_both = clip1_len + clip2_len + 10
    if after_both < result.shape[1]:
        assert abs(result[0, after_both]) < 1e-5

    print("  PASS: test_assemble_timeline_no_overlap")


def test_node_input_types():
    """Test that the node class INPUT_TYPES is properly structured."""
    from nodes import IndexTTS2ScriptDubbing

    inputs = IndexTTS2ScriptDubbing.INPUT_TYPES()
    assert "required" in inputs
    assert "optional" in inputs

    req = inputs["required"]
    assert "model" in req
    assert "script_srt" in req
    assert "emo_alpha" in req

    opt = inputs["optional"]
    assert "emo_audio_prompt" in opt, "emo_audio_prompt should be optional"
    for i in range(1, 8):
        assert f"voice_{i}" in opt, f"Missing voice_{i}"
        assert f"voice_{i}_name" in opt, f"Missing voice_{i}_name"

    assert IndexTTS2ScriptDubbing.RETURN_TYPES == ("AUDIO",)
    assert IndexTTS2ScriptDubbing.FUNCTION == "generate"
    assert IndexTTS2ScriptDubbing.CATEGORY == "audio/IndexTTS2"

    print("  PASS: test_node_input_types")


if __name__ == "__main__":
    print("Running ScriptDubbing unit tests...\n")

    test_parse_srt_standard()
    test_parse_srt_english_colon()
    test_parse_srt_with_emotion()
    test_parse_srt_empty()
    test_parse_srt_timestamps()
    test_split_character_dialogue()
    test_extract_emotion_segment()
    test_assemble_timeline()
    test_assemble_timeline_empty()
    test_assemble_timeline_no_overlap()
    test_node_input_types()

    print("\nAll tests passed!")
