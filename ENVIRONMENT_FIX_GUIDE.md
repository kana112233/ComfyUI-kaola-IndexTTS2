# 🛠️ 环境故障排查与修复总结报告

## 1. 现象描述
在 RTX 5090 服务器上安装 `index-tts` 和相关 ComfyUI 插件时，曾遭遇以下持续性报错：
- **Protobuf 冲突**：提示 `cannot import name 'builder' from 'google.protobuf.internal'`。
- **CUDA 算子失效**：`torchvision::nms` 缺失或 `Flash-Attention` 编译报错。
- **段错误 (Segment Fault)**：程序启动即崩溃。

## 2. 根源分析：环境幽灵 (Environment Shadowing)
经过深深度排查，发现系统内同时并存了两个 Python 环境管理器：
1.  **Pyenv (全局)**: 固定在 Python 3.10。
2.  **Conda (私有环境)**: 创建并激活了 Python 3.11。

### 为什么这种并存是致命的？
*   **指令劫持 (Shim Overlapping)**：Pyenv 通过在 `$PATH` 前端放置 `shims` 来拦截命令。即便激活了 Conda 环境，执行 `pip` 或 `python` 时仍可能命中 Pyenv 的 3.10 路径。
*   **二进制编译错位 (ABI Mismatch)**：
    *   `index-tts` 需要编译 C++/CUDA 扩展（如 Flash-Attention）。
    *   编译工具由于 Pyenv 的干扰，错误地引用了 **3.10 的 Python 头文件** 来编译 **3.11 的运行库**。
    *   这种“弗兰肯斯坦”式的混合产生了无法通过补丁修复的底层二进制冲突，直接导致显卡算子调用失败。

## 3. 最终解决方案：环境归一化
**关键动作：彻底移除 Pyenv，只保留 Conda。**

### 修复后的状态：
*   **路径纯净**：系统只识别 Conda 作为唯一的环境管理器。
*   **编译一致性**：`pip install -e .` 现在能准确识别当前 Conda 环境的 3.11 头文件和 CUDA 12.8 驱动，生成的二进制完全适配 RTX 5090 (sm_100)。
*   **库加载正常**：Protobuf 等动态库的搜索路径不再受 Pyenv 全局变量干扰。

## 4. 给 AI 开发者（及 RTX 5090 用户）的建议
1.  **禁忌**：严禁在同一台深度学习服务器上混用 `Pyenv` 和 `Conda`。
2.  **首选 Conda**：对于涉及 GPU 驱动、CUDA 算子编译的项目，Conda 处理二进制依赖的能力远强于 Pyenv。
3.  **环境隔离**：每次启动 ComfyUI 前，务必确认 `which python` 指向的是您预期的 Conda 环境路径。

---
*本报告由 Antigravity 整理，记录于 2026年2月1日，针对 IndexTTS-2 集成过程中的环境冲突。*
