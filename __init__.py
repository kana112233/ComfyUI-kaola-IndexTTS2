"""
ComfyUI-kaola-IndexTTS2
ComfyUI custom nodes for IndexTTS-2 zero-shot text-to-speech synthesis
"""

import os
import sys

# Add node directory to sys.path so that kaola_indextts internal absolute
# imports (e.g. "from kaola_indextts.gpt.model_v2 import ...") can resolve.
_node_dir = os.path.dirname(os.path.abspath(__file__))
if _node_dir not in sys.path:
    sys.path.insert(0, _node_dir)

from .nodes import (
    IndexTTS2ModelLoader,
    IndexTTS2VoiceClone,
    IndexTTS2EmotionAudio,
    IndexTTS2EmotionVector,
    IndexTTS2EmotionText,
    IndexTTS2ScriptDubbing,
)

NODE_CLASS_MAPPINGS = {
    "IndexTTS2ModelLoader": IndexTTS2ModelLoader,
    "IndexTTS2VoiceClone": IndexTTS2VoiceClone,
    "IndexTTS2EmotionAudio": IndexTTS2EmotionAudio,
    "IndexTTS2EmotionVector": IndexTTS2EmotionVector,
    "IndexTTS2EmotionText": IndexTTS2EmotionText,
    "IndexTTS2ScriptDubbing": IndexTTS2ScriptDubbing,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IndexTTS2ModelLoader": "IndexTTS2 Model Loader",
    "IndexTTS2VoiceClone": "IndexTTS2 Voice Clone",
    "IndexTTS2EmotionAudio": "IndexTTS2 Emotion (Audio)",
    "IndexTTS2EmotionVector": "IndexTTS2 Emotion (Vector)",
    "IndexTTS2EmotionText": "IndexTTS2 Emotion (Text)",
    "IndexTTS2ScriptDubbing": "IndexTTS2 Script Dubbing (SRT)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
