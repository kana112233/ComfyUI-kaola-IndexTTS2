from .nodes.indextts2_node import IndexTTS2Simple
from .nodes.indextts2_node_advanced import IndexTTS2Advanced
from .nodes.indextts2_node_emovec import IndexTTS2EmotionVector
from .nodes.indextts2_node_emotext import IndexTTS2EmotionFromText
from .nodes.indextts2_save_audio import IndexTTS2SaveAudio

NODE_CLASS_MAPPINGS = {
    "IndexTTS2Advanced": IndexTTS2Advanced,
    "IndexTTS2EmotionFromText": IndexTTS2EmotionFromText,
    "IndexTTS2EmotionVector": IndexTTS2EmotionVector,
    "IndexTTS2SaveAudio": IndexTTS2SaveAudio,
    "IndexTTS2Simple": IndexTTS2Simple,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IndexTTS2Advanced": "IndexTTS2 Advanced",
    "IndexTTS2EmotionFromText": "IndexTTS2 Emotion From Text",
    "IndexTTS2EmotionVector": "IndexTTS2 Emotion Vector",
    "IndexTTS2SaveAudio": "IndexTTS2 Save Audio",
    "IndexTTS2Simple": "IndexTTS2 Simple",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
