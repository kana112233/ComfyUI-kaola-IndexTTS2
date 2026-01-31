# ComfyUI Registry 发布指南

## ❌ 当前错误

```
Failed to publish node version: 400 {"message":"Failed to validate token"}
```

**原因**: GitHub Secrets 中的 `COMFYUI_REGISTRY_TOKEN` 未配置或无效。

## ✅ 解决方案

### 步骤 1: 获取 ComfyUI Registry Token

1. 访问 [ComfyUI Registry](https://registry.comfy.org/)
2. 使用 GitHub 账号登录
3. 进入个人设置 (Settings)
4. 生成 Personal Access Token
5. 复制 token（只会显示一次！）

### 步骤 2: 添加到 GitHub Secrets

1. 打开你的 GitHub 仓库
   ```
   https://github.com/kana112233/ComfyUI-kaola-IndexTTS2
   ```

2. 进入 **Settings** → **Secrets and variables** → **Actions**

3. 点击 **New repository secret**

4. 添加 secret:
   - **Name**: `COMFYUI_REGISTRY_TOKEN`
   - **Value**: 粘贴你的 token

5. 点击 **Add secret**

### 步骤 3: 重新触发发布

有两种方式：

**方式 1: 创建新的 Release**
```bash
git tag v1.0.0
git push origin v1.0.0
```

**方式 2: 手动触发 Workflow**
1. 进入 **Actions** 标签
2. 选择 **Publish to ComfyUI Registry**
3. 点击 **Run workflow**

## 📋 发布检查清单

- ✅ `pyproject.toml` 配置完成
- ✅ `.github/workflows/publish.yml` 已创建
- ❌ `COMFYUI_REGISTRY_TOKEN` 需要配置
- ⏳ 等待发布成功

## 🔍 验证发布

发布成功后，你的节点会出现在：
```
https://registry.comfy.org/nodes/comfyui-kaola-indextts2
```

用户可以通过 ComfyUI Manager 安装：
```
搜索: "IndexTTS-2" 或 "kaola-indextts2"
```

## 📝 注意事项

1. **Token 安全**: 
   - 不要在代码中硬编码 token
   - 只通过 GitHub Secrets 使用

2. **版本号**: 
   - 每次发布需要更新 `pyproject.toml` 中的版本号
   - 遵循语义化版本 (Semantic Versioning)

3. **依赖**: 
   - 确保 `requirements.txt` 包含所有依赖
   - IndexTTS-2 核心库需要用户手动安装

## 🚀 可选：手动发布

如果不想使用 GitHub Actions，也可以手动发布：

```bash
# 安装 ComfyUI Registry CLI
pip install comfyui-registry

# 登录
comfyui-registry login

# 发布
comfyui-registry publish
```

## 💡 当前状态

- ✅ 代码已完成并推送到 GitHub
- ✅ 所有功能已测试
- ✅ 文档已完善
- ⏳ 等待配置 Registry Token 后发布

**不配置 token 也可以使用！** 用户可以通过以下方式安装：

1. **ComfyUI Manager** - 从 GitHub URL 安装
2. **手动安装** - 克隆仓库到 `custom_nodes/`
3. **Git 子模块** - 添加为子模块
