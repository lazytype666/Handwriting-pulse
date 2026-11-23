# 本地数学强化小模型 & 视觉识别服务配置指南 (`ml/local_llm/`)

本目录用于将本地**数学强化推理小模型**与**多模态视觉公式 OCR 模型**快速接入《笔迹脉搏》端侧引擎，打通 **“拍照识题（Markdown + LaTeX）→ 预览确认 → 后台静默生成解题图谱 → 写入 `SolutionCache`（答案封存）”** 全链路闭环。

## 1. 本地模型部署清单（基于 Ollama）

在装有 NVIDIA GPU 或大内存的电脑 / 平板局域网主机运行：

```sh
# 1) 安装本地数学强化推理基座（Qwen2-Math-1.5B，约 934 MB；响应极快且直接输出步骤与 \boxed{}）
ollama pull qwen2-math:1.5b

# （可选对照）深度慢思考蒸馏基座（DeepSeek-R1-Distill-Qwen-1.5B，约 1.1 GB，需较大 token 预算）
ollama pull deepseek-r1:1.5b

# 2) 按本目录 Modelfile 导入专为《笔迹脉搏》调优参数的本地数学小模型
ollama create inkpulse-math -f ml/local_llm/Modelfile

# 3)（可选）拉取本地视觉公式识题模型，配合或增强手机端 ML Kit 识题（约 3.2 GB）
ollama pull qwen2.5vl:3b
```

## 2. 移动端 `.env.local` 配置（真机局域网直连）

在 Windows 上先让 Ollama 监听局域网（仅开发联调开启）：

```powershell
$env:OLLAMA_HOST="0.0.0.0:11434"
ollama serve
```

在仓库根目录复制 `.env.example` 为 `.env.local`：
- **推荐开发方式（手机开热点连电脑 / 动态 DHCP）**：填 `http://auto:11434`，App 启动时会自动从 Expo Metro 解析当前电脑热点/局域网 IP（Node 单测时自动回退 `127.0.0.1`）；
- **固定局域网 IP 方式**：把 `<电脑局域网IP>` 替换为本机 IPv4 地址（例如 `192.168.1.25`）：

```env
EXPO_PUBLIC_EDGE_LLM_URL=http://auto:11434
EXPO_PUBLIC_EDGE_LLM_MODEL=inkpulse-math
EXPO_PUBLIC_EDGE_LLM_TIMEOUT_MS=45000

# （可选）如需使用视觉多模态直接转 Markdown + LaTeX：
EXPO_PUBLIC_VLM_BASE_URL=http://auto:11434
EXPO_PUBLIC_VLM_MODEL=qwen2.5vl:3b
```
