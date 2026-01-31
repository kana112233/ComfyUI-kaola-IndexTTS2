# IndexTTS-2 ComfyUI 节点 - 示例工作流

本目录包含所有 IndexTTS-2 节点的示例工作流。

## 工作流列表

### 1. 基础语音克隆 (`01_voice_clone.json`)
- **节点**: IndexTTS2ModelLoader + IndexTTS2VoiceClone + IndexTTS2SaveAudio
- **功能**: 使用参考音频进行零样本语音克隆
- **用途**: 最基础的 TTS 功能，克隆任何说话人的声音

### 2. 音频情感控制 (`02_emotion_audio.json`)
- **节点**: IndexTTS2ModelLoader + IndexTTS2EmotionAudio + IndexTTS2SaveAudio
- **功能**: 使用情感参考音频控制生成语音的情感
- **用途**: 通过提供带有特定情感的音频样本来控制输出情感
- **参数**: 
  - `emo_alpha`: 情感强度 (0.0-1.0)

### 3. 向量情感控制 (`03_emotion_vector.json`)
- **节点**: IndexTTS2ModelLoader + IndexTTS2EmotionVector + IndexTTS2SaveAudio
- **功能**: 使用8维情感向量精确控制情感
- **用途**: 精确控制情感的各个维度
- **情感维度**:
  - happy (快乐)
  - angry (愤怒)
  - sad (悲伤)
  - afraid (恐惧)
  - disgusted (厌恶)
  - melancholic (忧郁)
  - surprised (惊讶)
  - calm (平静)

### 4. 文本情感控制 (`04_emotion_text.json`)
- **节点**: IndexTTS2ModelLoader + IndexTTS2EmotionText + IndexTTS2SaveAudio
- **功能**: 使用自然语言描述控制情感
- **用途**: 最用户友好的情感控制方式
- **示例**: "happy and excited", "sad and melancholic", "angry and frustrated"

## 使用说明

1. **导入工作流**
   - 在 ComfyUI 中点击 "Load" 按钮
   - 选择对应的 JSON 文件

2. **配置路径**
   - 将 `path/to/speaker.wav` 替换为实际的说话人音频路径
   - 将 `path/to/emotion.wav` 替换为实际的情感参考音频路径（仅用于 02）

3. **设置环境变量** (重要!)
   ```bash
   export W2V_BERT_PATH="/path/to/models/w2v-bert-2.0"
   ```

4. **运行工作流**
   - 点击 "Queue Prompt" 开始生成

## 模型路径

确保模型文件位于正确位置：
- IndexTTS-2: `ComfyUI/models/IndexTTS-2/`
- wav2vec2bert: 通过 `W2V_BERT_PATH` 环境变量指定

## 性能参考

- **设备**: Apple Silicon (MPS) / NVIDIA GPU (CUDA) / CPU
- **RTF**: ~4-13 (取决于设备)
- **音频质量**: 24kHz, 16-bit

## 提示

- 使用 `use_random=false` 获得确定性输出
- 调整 `emo_alpha` 控制情感强度
- 情感向量各维度总和建议为 1.0
- 文本情感描述支持中英文
