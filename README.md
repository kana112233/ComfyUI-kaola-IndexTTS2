# ComfyUI-kaola-IndexTTS2

Lightweight ComfyUI wrapper for IndexTTS 2 (voice cloning + emotion control). Nodes call the upstream inference code so behavior stays matched with the original repo.

Original repo: [https://github.com/index-tts/index-tts](https://github.com/index-tts/index-tts)

## Features

* **Voice Cloning**: Clone any voice from a short audio sample
* **Emotion Control**: Fine-tune emotional expression in generated speech
* **Advanced Generation**: Control sampling parameters, speech speed, and more
* **FP16 Support**: Optional FP16 mode for faster inference on CUDA devices
* **Audio Output**: Save audio in WAV or MP3 format with customizable quality

## Install

1. Clone this repository into `ComfyUI/custom_nodes/`:
   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/kana112233/ComfyUI-kaola-IndexTTS2.git
   ```

2. Install dependencies inside your ComfyUI Python environment:
   ```bash
   cd ComfyUI-kaola-IndexTTS2
   pip install wetext
   pip install -r requirements.txt
   ```

## Models

* Create `checkpoints/` in the repo root and copy the IndexTTS-2 release there ([https://huggingface.co/IndexTeam/IndexTTS-2/tree/main](https://huggingface.co/IndexTeam/IndexTTS-2/tree/main))
* Missing files will be cached from Hugging Face automatically

## Nodes

### IndexTTS2 Simple
Basic text-to-speech generation node with speaker audio, text input, and optional emotion control.

**Inputs:**
* `audio` - Speaker audio sample for voice cloning
* `text` - Text to synthesize
* `emotion_control_weight` - Weight for emotion influence (0.0-1.0)
* `emotion_audio` (optional) - Separate audio for emotion reference
* `emotion_vector` (optional) - Pre-computed emotion vector
* `use_fp16` (optional) - Enable FP16 precision (CUDA only)
* `output_gain` (optional) - Output volume multiplier (0.0-4.0)

**Outputs:**
* Audio tensor
* Status string

### IndexTTS2 Advanced
Extended version with full control over generation parameters.

**Additional Inputs:**
* `use_random_style` - Use random emotion preset
* `interval_silence_ms` - Silence between segments (ms)
* `max_text_tokens_per_segment` - Maximum tokens per segment
* `seed` - Random seed for reproducibility (-1 for random)
* `do_sample` - Enable sampling vs beam search
* `temperature` - Sampling temperature
* `top_p` - Nucleus sampling parameter
* `top_k` - Top-k sampling parameter
* `repetition_penalty` - Penalty for repetition
* `length_penalty` - Penalty for length
* `num_beams` - Number of beams for beam search
* `max_mel_tokens` - Maximum mel spectrogram tokens
* `typical_sampling` - Enable typical sampling
* `typical_mass` - Typical sampling mass
* `speech_speed` - Speech speed multiplier (0.25-4.0)

### IndexTTS2 Emotion Vector
Create emotion vectors using eight slider controls.

**Inputs (all 0.0-1.4, sum must be ≤ 1.5):**
* `happy`
* `angry`
* `sad`
* `afraid`
* `disgusted`
* `melancholic`
* `surprised`
* `calm`

**Output:**
* Emotion vector for use with generation nodes

### IndexTTS2 Emotion From Text
Generate emotion vectors from descriptive text (requires ModelScope and QwenEmotion model).

**Input:**
* `text` - Descriptive emotion text

**Outputs:**
* Emotion vector
* Summary string

### IndexTTS2 Save Audio
Save generated audio to disk with format options and inline preview.

**Inputs:**
* `audio` - Audio tensor to save
* `name` - Filename prefix
* `format` - Output format (wav/mp3)
* `normalize_peak` (optional) - Normalize to ~0.98 peak
* `wav_pcm` (optional) - WAV bit depth (pcm16/pcm24/f32)
* `mp3_bitrate` (optional) - MP3 quality (128k/192k/256k/320k)

## Examples

### Basic Voice Cloning
```
Speaker Audio -> IndexTTS2 Simple -> Save Audio
```

### Emotion-Controlled Speech
```
Speaker Audio + Emotion Audio -> IndexTTS2 Simple -> Save Audio
```

### Custom Emotion Vector
```
Emotion Vector -> IndexTTS2 Simple -> Save Audio
```

### Text-Based Emotion
```
Emotion From Text -> IndexTTS2 Simple -> Save Audio
```

## Troubleshooting

* **Windows Only**: Currently tested primarily on Windows; DeepSpeed is disabled
* **wetext Module**: Install `wetext` if missing: `pip install wetext`
* **Emotion Vector Sum**: The sum of all emotion values must be ≤ 1.5
* **CUDA/FP16**: FP16 mode only works on CUDA devices
* **Model Files**: Download model files from HuggingFace if they don't auto-download

## Updates

* Initial release with full node support for IndexTTS2
* Support for FP32/FP16 precision modes
* Output gain control
* Advanced generation parameters
* Save Audio with WAV/MP3 format options

## License

This is a wrapper for IndexTTS2. Please refer to the original IndexTTS repository for licensing information.

## Credits

* Original IndexTTS2: [https://github.com/index-tts/index-tts](https://github.com/index-tts/index-tts)
* Reference implementation: [https://github.com/snicolast/ComfyUI-IndexTTS2](https://github.com/snicolast/ComfyUI-IndexTTS2)
