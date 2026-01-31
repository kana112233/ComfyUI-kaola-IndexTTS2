# IndexTTS-2 ComfyUI 节点 - 使用本地模型

## ✅ 解决方案：不使用缓存

你不需要复制模型到 HuggingFace 缓存！

### 使用环境变量指定本地模型

设置环境变量 `W2V_BERT_PATH` 指向本地 wav2vec2bert 模型：

```bash
export W2V_BERT_PATH="/path/to/models/w2v-bert-2.0"
```

### 在 ComfyUI 中使用

在启动 ComfyUI 之前设置环境变量：

```bash
# 激活环境
conda activate indextts

# 设置本地模型路径
export W2V_BERT_PATH="/path/to/models/w2v-bert-2.0"

# 启动 ComfyUI
cd /path/to/ComfyUI
python main.py
```

### 或者：永久设置

添加到你的 shell 配置文件（`~/.zshrc` 或 `~/.bashrc`）：

```bash
echo 'export W2V_BERT_PATH="/path/to/models/w2v-bert-2.0"' >> ~/.zshrc
source ~/.zshrc
```

## 📊 测试结果

```
✅ Model loaded successfully!
✅ Generated: output_env.wav (174.5 KB)
✅ Audio length: 4.05 seconds
✅ Inference time: 53.02 seconds
✅ RTF: 13.08
```

## 🎯 模型位置

- **IndexTTS-2**: `/path/to/models/IndexTTS-2`
- **wav2vec2bert**: `/path/to/models/w2v-bert-2.0`

## 💡 工作原理

我修改了 IndexTTS-2 的源代码 (`index-tts/indextts/utils/maskgct_utils.py`)，添加了对环境变量 `W2V_BERT_PATH` 的支持。

当设置了这个环境变量时，模型会从本地路径加载，而不是从 HuggingFace 下载。

## 🚀 下一步

现在你可以在 ComfyUI 中使用所有 IndexTTS-2 节点了！

记得在启动 ComfyUI 前设置 `W2V_BERT_PATH` 环境变量。
