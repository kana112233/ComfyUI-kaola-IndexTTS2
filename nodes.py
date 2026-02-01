"""
IndexTTS-2 ComfyUI Nodes
Provides nodes for zero-shot text-to-speech synthesis with emotion control
"""

import os
import re
import sys
import tempfile
import threading

import wave

import torch
import numpy as np
import folder_paths

# ---------------------------------------------------------------------------
# Model cache: avoid reloading the heavy model on every execution
# ---------------------------------------------------------------------------
_MODEL_CACHE = {}
_CACHE_LOCK = threading.RLock()


def _get_model(config_path, model_dir, device, use_fp16, use_cuda_kernel, use_deepspeed):
    """Return a cached IndexTTS2 instance or create a new one."""
    from indextts.infer_v2 import IndexTTS2

    key = (
        os.path.abspath(config_path),
        os.path.abspath(model_dir),
        str(device),
        bool(use_fp16),
        bool(use_cuda_kernel),
        bool(use_deepspeed),
    )

    with _CACHE_LOCK:
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]

    model = IndexTTS2(
        cfg_path=config_path,
        model_dir=model_dir,
        device=device,
        use_fp16=use_fp16,
        use_cuda_kernel=use_cuda_kernel,
        use_deepspeed=use_deepspeed,
    )

    with _CACHE_LOCK:
        existing = _MODEL_CACHE.get(key)
        if existing is not None:
            return existing
        _MODEL_CACHE[key] = model

    return model


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _save_wav(path, wav_cn, sr):
    """Save float32 numpy waveform (channels, samples) to WAV PCM16.
    Uses soundfile if available, otherwise falls back to the wave stdlib.
    """
    wav_cn = np.clip(wav_cn, -1.0, 1.0)
    pcm = (wav_cn * 32767.0).astype(np.int16)

    try:
        import soundfile as sf
        # soundfile expects (samples, channels)
        sf.write(path, pcm.T if pcm.ndim == 2 else pcm, sr, subtype="PCM_16")
        return
    except Exception:
        pass

    # Fallback: stdlib wave module
    n_channels = 1 if pcm.ndim == 1 else pcm.shape[0]
    n_frames = pcm.shape[-1]
    interleaved = pcm.T.tobytes() if pcm.ndim == 2 else pcm.tobytes()
    with wave.open(path, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(interleaved)


def _audio_to_temp_file(audio):
    """Save ComfyUI AUDIO dict to a temporary wav file and return the path."""
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    if waveform.dim() == 3:
        waveform = waveform.squeeze(0)
    # Convert to numpy float32 (channels, samples)
    wav = waveform.cpu().numpy().astype(np.float32)
    if np.abs(wav).max() > 1.0:
        wav = wav / 32767.0  # int16 range

    fd, path = tempfile.mkstemp(suffix=".wav", prefix="indextts2_")
    os.close(fd)
    _save_wav(path, wav, int(sample_rate))
    return path


def _result_to_audio(result):
    """Convert IndexTTS2 infer result (sr, wav_data) to ComfyUI AUDIO dict."""
    sr, wav_data = result
    if hasattr(wav_data, "cpu"):
        wav_data = wav_data.cpu().numpy()
    wav = np.asarray(wav_data)

    # Normalize to float32 [-1, 1]
    if np.issubdtype(wav.dtype, np.integer):
        wav = wav.astype(np.float32) / 32767.0
    else:
        wav = wav.astype(np.float32)

    # Ensure (channels, samples) shape
    if wav.ndim == 1:
        wav = wav[np.newaxis, :]
    elif wav.ndim == 2 and wav.shape[1] <= wav.shape[0]:
        # infer returns (samples, channels), transpose to (channels, samples)
        wav = wav.T

    waveform = torch.from_numpy(wav).unsqueeze(0)  # (B=1, C, N)
    return {"waveform": waveform, "sample_rate": int(sr)}


def _cleanup(*paths):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.unlink(p)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# SRT / Script Dubbing helpers
# ---------------------------------------------------------------------------

def _parse_srt(text):
    """Parse SRT formatted text into a list of entries.

    Supports both standard multi-line SRT and compact single-line SRT.
    Each entry is a dict with keys: index, start_ms, end_ms, character, dialogue.
    """

    def _ts_to_ms(ts):
        """Convert SRT timestamp (HH:MM:SS,mmm) to milliseconds."""
        ts = ts.strip().replace(",", ".")
        parts = ts.split(":")
        h, m = int(parts[0]), int(parts[1])
        s_parts = parts[2].split(".")
        s = int(s_parts[0])
        ms = int(s_parts[1]) if len(s_parts) > 1 else 0
        return h * 3600000 + m * 60000 + s * 1000 + ms

    entries = []
    lines = text.strip().splitlines()

    # Try standard multi-line SRT first:
    # pattern: index line, timestamp line, content lines, blank line separator
    timestamp_re = re.compile(
        r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})"
    )

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Check if this is an index line (pure number) followed by a timestamp line
        if line.isdigit() and i + 1 < len(lines):
            ts_match = timestamp_re.search(lines[i + 1])
            if ts_match:
                index = int(line)
                start_ms = _ts_to_ms(ts_match.group(1))
                end_ms = _ts_to_ms(ts_match.group(2))
                # Collect content lines until blank line or next index
                i += 2
                content_lines = []
                while i < len(lines) and lines[i].strip():
                    content_lines.append(lines[i].strip())
                    i += 1
                content = " ".join(content_lines)
                character, dialogue = _split_character_dialogue(content)
                entries.append({
                    "index": index,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "character": character,
                    "dialogue": dialogue,
                })
                continue

        # Try compact single-line format: "index | timestamp --> timestamp | character: dialogue"
        # or timestamp embedded in the line
        ts_match = timestamp_re.search(line)
        if ts_match:
            start_ms = _ts_to_ms(ts_match.group(1))
            end_ms = _ts_to_ms(ts_match.group(2))
            # Extract content after the timestamp
            after_ts = line[ts_match.end():].strip()
            if after_ts.startswith("|"):
                after_ts = after_ts[1:].strip()
            # Extract index before the timestamp
            before_ts = line[:ts_match.start()].strip()
            index = len(entries) + 1
            if before_ts:
                idx_part = before_ts.rstrip("|").strip()
                if idx_part.isdigit():
                    index = int(idx_part)
            character, dialogue = _split_character_dialogue(after_ts)
            if dialogue:
                entries.append({
                    "index": index,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "character": character,
                    "dialogue": dialogue,
                })
        i += 1

    return entries


def _split_character_dialogue(content):
    """Split 'Character: dialogue' into (character, dialogue).

    Supports both Chinese colon and English colon.
    Returns ("", content) if no character prefix is found.
    """
    # Match "角色名：台词" or "角色名: 台词"
    m = re.match(r"^([^:：]+?)\s*[：:]\s*(.+)$", content, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", content.strip()


def _extract_emotion_segment(emo_audio, start_ms, end_ms):
    """Extract a time segment from the emotion reference audio.

    Returns (audio_dict, info_str) where info_str describes what happened.
    Falls back to the full audio if the segment is too short (<0.1s) or out of range.
    """
    waveform = emo_audio["waveform"]  # (B, C, N)
    sr = emo_audio["sample_rate"]

    if waveform.dim() == 3:
        wav = waveform[0]  # (C, N)
    else:
        wav = waveform

    total_samples = wav.shape[-1]
    total_ms = total_samples * 1000 / sr
    start_sample = int(start_ms * sr / 1000)
    end_sample = int(end_ms * sr / 1000)

    min_samples = int(0.1 * sr)  # 0.1 seconds

    if start_sample >= total_samples:
        info = f"fallback(全部 {total_ms:.0f}ms): 起始{start_ms}ms超出音频长度"
        return emo_audio, info

    if end_sample <= start_sample or (end_sample - start_sample) < min_samples:
        info = f"fallback(全部 {total_ms:.0f}ms): 片段太短({end_ms - start_ms}ms)"
        return emo_audio, info

    end_sample = min(end_sample, total_samples)
    actual_ms = (end_sample - start_sample) * 1000 / sr
    segment = wav[:, start_sample:end_sample]
    info = f"切片 {start_ms}~{end_ms}ms (实际{actual_ms:.0f}ms)"
    return {
        "waveform": segment.unsqueeze(0),  # (1, C, N)
        "sample_rate": sr,
    }, info


def _assemble_timeline(audio_segments, srt_entries, output_sr):
    """Assemble synthesized audio segments onto a timeline based on SRT timestamps.

    audio_segments: list of (srt_entry, audio_numpy) where audio_numpy is (channels, samples)
    srt_entries: the full list of parsed SRT entries (for computing total duration)
    output_sr: sample rate of the output

    Strategy: each segment is placed at max(srt_start, previous_segment_end),
    so segments never overlap and are never truncated — they just push the
    timeline forward naturally when they run longer than the SRT slot.

    Returns numpy array of shape (1, total_samples).
    """
    if not audio_segments:
        raise RuntimeError("No audio segments to assemble")

    # Sort by start time
    audio_segments.sort(key=lambda x: x[0]["start_ms"])

    # First pass: compute actual placement positions and total length
    placements = []  # (start_sample, mono_segment)
    cursor = 0  # tracks the end of the last placed segment (in samples)

    for entry, audio_np in audio_segments:
        srt_start = int(entry["start_ms"] * output_sr / 1000)

        # Ensure mono
        seg = audio_np[0] if audio_np.ndim == 2 else audio_np

        # Place at whichever is later: SRT start or end of previous segment
        start_sample = max(srt_start, cursor)
        end_sample = start_sample + len(seg)
        cursor = end_sample

        placements.append((start_sample, seg))

    # Allocate buffer
    total_samples = cursor + int(0.5 * output_sr)  # small tail
    buffer = np.zeros((1, total_samples), dtype=np.float32)

    for start_sample, seg in placements:
        buffer[0, start_sample:start_sample + len(seg)] = seg

    # Trim trailing silence
    abs_buf = np.abs(buffer[0])
    nonzero = np.nonzero(abs_buf > 1e-6)[0]
    if len(nonzero) > 0:
        tail = int(0.5 * output_sr)
        trim_end = min(nonzero[-1] + tail, total_samples)
        buffer = buffer[:, :trim_end]

    return buffer


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

class IndexTTS2ModelLoader:
    """Load IndexTTS-2 model with configurable optimization settings"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_dir": ("STRING", {
                    "default": "IndexTTS-2",
                    "multiline": False,
                }),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "use_fp16": ("BOOLEAN", {"default": False}),
                "use_cuda_kernel": ("BOOLEAN", {"default": False}),
                "use_deepspeed": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("INDEXTTS2_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "audio/IndexTTS2"

    def load_model(self, model_dir, device, use_fp16, use_cuda_kernel, use_deepspeed):
        # Resolve model path
        if not os.path.isabs(model_dir):
            full_path = os.path.join(folder_paths.models_dir, model_dir)
            if not os.path.exists(full_path):
                full_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), model_dir
                )
            model_dir = full_path

        config_path = os.path.join(model_dir, "config.yaml")
        if not os.path.isdir(model_dir) or not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"Model not found at: {model_dir}\n"
                f"Download with:\n"
                f"  huggingface-cli download IndexTeam/IndexTTS-2 --local-dir={model_dir}"
            )

        # Resolve device
        if device == "auto":
            if torch.cuda.is_available():
                resolved_device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                resolved_device = "mps"
            else:
                resolved_device = "cpu"
        else:
            resolved_device = device

        # Logic for optimizations: only enable if hardware supports it
        # and user has it checked. 
        if resolved_device != "cuda":
            if use_cuda_kernel:
                print(f"Warning: CUDA Kernel is only supported on CUDA devices. Disabling for {resolved_device}.")
                use_cuda_kernel = False
            if use_deepspeed:
                print(f"Warning: DeepSpeed is typically used on CUDA. Proceed with caution on {resolved_device}.")

        print(f"Loading IndexTTS-2 | Device: {resolved_device} | FP16: {use_fp16} | CUDA Kernel: {use_cuda_kernel} | DeepSpeed: {use_deepspeed}")

        print(f"Loading IndexTTS-2 from: {model_dir}")

        model = _get_model(
            config_path=config_path,
            model_dir=model_dir,
            device=resolved_device,
            use_fp16=use_fp16,
            use_cuda_kernel=use_cuda_kernel,
            use_deepspeed=use_deepspeed,
        )

        return (model,)


class IndexTTS2VoiceClone:
    """Basic voice cloning with single reference audio"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("INDEXTTS2_MODEL",),
                "text": ("STRING", {"default": "Hello, this is a test of voice cloning with IndexTTS-2.", "multiline": True}),
                "spk_audio_prompt": ("AUDIO",),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "use_random": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/IndexTTS2"

    def generate(self, model, text, spk_audio_prompt, use_random, **kwargs):
        spk_path = _audio_to_temp_file(spk_audio_prompt)
        try:
            result = model.infer(
                spk_audio_prompt=spk_path,
                text=text,
                output_path=None,
                use_random=use_random,
                temperature=kwargs.get("temperature", 1.0),
                top_k=kwargs.get("top_k", 0),
                top_p=kwargs.get("top_p", 1.0),
                verbose=True,
            )
            return (_result_to_audio(result),)
        finally:
            _cleanup(spk_path)


class IndexTTS2EmotionAudio:
    """Voice synthesis with separate emotion reference audio"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("INDEXTTS2_MODEL",),
                "text": ("STRING", {"default": "This speech has a specific emotional tone from the reference audio.", "multiline": True}),
                "spk_audio_prompt": ("AUDIO",),
                "emo_audio_prompt": ("AUDIO",),
                "emo_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "use_random": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/IndexTTS2"

    def generate(self, model, text, spk_audio_prompt, emo_audio_prompt, emo_alpha, use_random, **kwargs):
        spk_path = _audio_to_temp_file(spk_audio_prompt)
        emo_path = _audio_to_temp_file(emo_audio_prompt)
        try:
            result = model.infer(
                spk_audio_prompt=spk_path,
                text=text,
                output_path=None,
                emo_audio_prompt=emo_path,
                emo_alpha=emo_alpha,
                use_random=use_random,
                temperature=kwargs.get("temperature", 1.0),
                top_k=kwargs.get("top_k", 0),
                top_p=kwargs.get("top_p", 1.0),
                verbose=True,
            )
            return (_result_to_audio(result),)
        finally:
            _cleanup(spk_path, emo_path)


class IndexTTS2EmotionVector:
    """Control emotions via 8-dimensional vector"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("INDEXTTS2_MODEL",),
                "text": ("STRING", {"default": "Wow! This is amazing!", "multiline": True}),
                "spk_audio_prompt": ("AUDIO",),
                "happy": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "angry": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "sad": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "afraid": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "disgusted": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "melancholic": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "surprised": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "calm": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "use_random": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/IndexTTS2"

    def generate(self, model, text, spk_audio_prompt, happy, angry, sad, afraid,
                 disgusted, melancholic, surprised, calm, use_random, **kwargs):
        spk_path = _audio_to_temp_file(spk_audio_prompt)
        emo_vector = [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
        try:
            result = model.infer(
                spk_audio_prompt=spk_path,
                text=text,
                output_path=None,
                emo_vector=emo_vector,
                use_random=use_random,
                temperature=kwargs.get("temperature", 1.0),
                top_k=kwargs.get("top_k", 0),
                top_p=kwargs.get("top_p", 1.0),
                verbose=True,
            )
            return (_result_to_audio(result),)
        finally:
            _cleanup(spk_path)


class IndexTTS2EmotionText:
    """Control emotions via natural language description"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("INDEXTTS2_MODEL",),
                "text": ("STRING", {"default": "Emotionally expressive speech.", "multiline": True}),
                "spk_audio_prompt": ("AUDIO",),
                "use_emo_text": ("BOOLEAN", {"default": True}),
                "emo_alpha": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "use_random": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "emo_text": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/IndexTTS2"

    def generate(self, model, text, spk_audio_prompt, use_random, **kwargs):
        spk_path = _audio_to_temp_file(spk_audio_prompt)
        try:
            inf_kwargs = {
                "spk_audio_prompt": spk_path,
                "text": text,
                "output_path": None,
                "use_emo_text": kwargs.get("use_emo_text", True),
                "emo_alpha": kwargs.get("emo_alpha", 0.6),
                "use_random": use_random,
                "temperature": kwargs.get("temperature", 1.0),
                "top_k": kwargs.get("top_k", 0),
                "top_p": kwargs.get("top_p", 1.0),
                "verbose": True,
            }
            emo_text = kwargs.get("emo_text", "")
            if emo_text and emo_text.strip():
                inf_kwargs["emo_text"] = emo_text

            result = model.infer(**inf_kwargs)
            return (_result_to_audio(result),)
        finally:
            _cleanup(spk_path)


class IndexTTS2ScriptDubbing:
    """Multi-character script dubbing driven by SRT subtitles.

    Parses an SRT script with character names, matches each line to a voice
    reference, uses time-aligned emotion audio segments, and assembles the
    synthesized audio onto the SRT timeline.
    """

    MAX_VOICES = 7

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "model": ("INDEXTTS2_MODEL",),
            "script_srt": ("STRING", {
                "default": (
                    "1\n"
                    "00:00:01,000 --> 00:00:03,000\n"
                    "唐僧：悟空，你又调皮了。\n"
                    "\n"
                    "2\n"
                    "00:00:04,000 --> 00:00:06,000\n"
                    "孙悟空：师父，俺老孙冤枉啊！\n"
                ),
                "multiline": True,
            }),
            "emo_audio_prompt": ("AUDIO",),
            "emo_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
            "top_k": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "use_random": ("BOOLEAN", {"default": False}),
        }

        optional = {}
        for i in range(1, cls.MAX_VOICES + 1):
            optional[f"voice_{i}"] = ("AUDIO",)
            optional[f"voice_{i}_name"] = ("STRING", {
                "default": "",
                "multiline": False,
            })

        return {"required": required, "optional": optional}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/IndexTTS2"

    def generate(self, model, script_srt, emo_audio_prompt, emo_alpha,
                 temperature, top_k, top_p, use_random, **kwargs):
        from comfy.utils import ProgressBar

        # --- Step 1: Parse SRT ---
        srt_entries = _parse_srt(script_srt)
        if not srt_entries:
            raise ValueError(
                "SRT 解析结果为空，请检查 script_srt 格式是否正确。"
            )
        print(f"[ScriptDubbing] Parsed {len(srt_entries)} SRT entries")
        pbar = ProgressBar(len(srt_entries))

        # --- Step 2: Build character -> voice AUDIO mapping ---
        voice_map = {}  # character_name -> AUDIO dict
        first_voice = None
        for i in range(1, self.MAX_VOICES + 1):
            voice_audio = kwargs.get(f"voice_{i}")
            voice_name = kwargs.get(f"voice_{i}_name", "").strip()
            if voice_audio is not None:
                if first_voice is None:
                    first_voice = voice_audio
                if voice_name:
                    voice_map[voice_name] = voice_audio

        if first_voice is None:
            raise ValueError(
                "至少需要连接一个音色参考音频 (voice_1 ~ voice_7)。"
            )

        # --- Step 3: Match each entry to a voice ---
        for entry in srt_entries:
            char_name = entry["character"]
            if char_name and char_name in voice_map:
                entry["_voice"] = voice_map[char_name]
            else:
                if char_name:
                    print(f"[ScriptDubbing] Warning: 角色 '{char_name}' 未匹配到音色，使用默认音色")
                entry["_voice"] = first_voice

        # --- Step 4: Synthesize each entry ---
        temp_files = []
        # Cache: same character voice -> reuse temp file path
        voice_file_cache = {}
        audio_segments = []  # list of (entry, audio_numpy)
        success_count = 0

        try:
            for idx, entry in enumerate(srt_entries):
                dialogue = entry["dialogue"]
                if not dialogue.strip():
                    continue

                char_name = entry["character"] or "_default_"

                # 4a: Voice reference temp file (reuse for same character)
                voice_audio = entry["_voice"]
                voice_id = id(voice_audio)
                if voice_id not in voice_file_cache:
                    spk_path = _audio_to_temp_file(voice_audio)
                    temp_files.append(spk_path)
                    voice_file_cache[voice_id] = spk_path
                spk_path = voice_file_cache[voice_id]

                # 4b: Emotion segment for this time range
                emo_segment, emo_info = _extract_emotion_segment(
                    emo_audio_prompt, entry["start_ms"], entry["end_ms"]
                )
                emo_path = _audio_to_temp_file(emo_segment)
                temp_files.append(emo_path)

                # 4c: Infer
                dial_preview = f"{dialogue[:30]}..." if len(dialogue) > 30 else dialogue
                print(
                    f"[ScriptDubbing] [{idx+1}/{len(srt_entries)}] "
                    f"角色={char_name}, 情绪={emo_info}, "
                    f"台词=\"{dial_preview}\""
                )

                try:
                    result = model.infer(
                        spk_audio_prompt=spk_path,
                        text=dialogue,
                        output_path=None,
                        emo_audio_prompt=emo_path,
                        emo_alpha=emo_alpha,
                        use_random=use_random,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        verbose=False,
                    )
                    audio_dict = _result_to_audio(result)
                    wav_np = audio_dict["waveform"].squeeze(0).cpu().numpy()
                    output_sr = audio_dict["sample_rate"]
                    audio_segments.append((entry, wav_np))
                    success_count += 1
                except Exception as e:
                    print(f"[ScriptDubbing] 第 {entry['index']} 条合成失败: {e}")
                finally:
                    pbar.update(1)

            if success_count == 0:
                raise RuntimeError("所有条目合成均失败，请检查模型和输入。")

            print(f"[ScriptDubbing] 合成完成: {success_count}/{len(srt_entries)} 条成功")

            # --- Step 5: Assemble timeline ---
            assembled = _assemble_timeline(audio_segments, srt_entries, output_sr)
            waveform = torch.from_numpy(assembled).unsqueeze(0)  # (1, 1, N)
            return ({"waveform": waveform, "sample_rate": output_sr},)

        finally:
            # --- Step 6: Cleanup ---
            _cleanup(*temp_files)
