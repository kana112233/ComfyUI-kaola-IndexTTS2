"""
IndexTTS-2 ComfyUI Nodes
Provides nodes for zero-shot text-to-speech synthesis with emotion control
"""

import os
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


