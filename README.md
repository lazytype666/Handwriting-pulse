# 笔迹脉搏 · Handwriting Pulse

笔迹脉搏是一张安卓数学草稿纸。学生可以粘贴一道题，边演算边书写。当停顿后的涂改或书写速度的明显下降触发规则时，草稿纸会给出一条思路，学生也可以继续请求更具体的提示。卡壳记录供本人回看，记录的是书写动作，无法据此断定学生是否掌握知识点。

## 在安卓真机上运行

电脑准备 Node.js 24.x（至少 24.3.0）与 npm，安卓手机或平板安装与 Expo SDK 58 匹配的 Expo Go。设备连接同一个可互访的 Wi-Fi。从仓库根目录启动本地开发服务，用 Expo Go 扫描终端二维码。[下载兼容的 Expo Go](https://expo.dev/go?sdkVersion=58&platform=android&device=true)

```sh
git clone https://github.com/lazytype666/Handwriting-pulse.git
cd Handwriting-pulse
npm install
npm run typecheck
npm start
```

初次运行即可试用本地提示。可以点「题干」，粘贴“等腰三角形的两条腰相等，已知顶角，怎样求底角？”，点「完成」返回画布。留空也可以书写、擦除、撤销和查看历史。

## 从书写到提示

1. 用「笔」写一行算式。抬笔停顿超过 3 秒，切换「橡皮」擦掉其中一个字，在停顿结束后的 3 秒内切回「笔」并在刚擦的区域落笔。底部会出现「停顿涂改」的 L1 提示。
2. 点「再提示一点」查看 L2，再点一次到 L3。L1 提醒可尝试的方向，L2 建议下一步，L3 串起推理路线。每层提示显示 8 秒后收起，也可手动收起。
3. 打开「历史」，看本次会话的事件时间、类型和两类触发次数。一次触发记录一条事件，升级提示不增加次数。返回画布继续书写。
4. 想体验另一条规则时，等待冷却结束，连续以较快速度画一段线，再保持落笔把移动速度大幅放慢至少 2 秒，观察「速度骤降」。默认阈值为近 1 秒书写速度低于个人近 30 秒平均速度的 30%，冷却时间为 10 秒。

「采样诊断」显示最近 1 秒真实收到的触点数和最近两点的时间差。展开实时特征可观察停顿、涂改、速度比和冷却剩余时间。在诊断详情拖动三个滑块可立即调整停顿阈值、速度比例与冷却时长。数据不足 30 秒时基线用已有观测时段计算。只停笔思考不会触发速度骤降。

## 启用针对题目的模型提示

选择提供 OpenAI 兼容 Chat Completions 的服务商，在其控制台创建演示密钥并查看可用模型。以 DeepSeek 为例，基础地址是 `https://api.deepseek.com`，模型名使用控制台当前提供的标识。[申请 DeepSeek API 密钥](https://platform.deepseek.com/api_keys) · [兼容地址与模型列表](https://api-docs.deepseek.com/)

将仓库根目录的 `.env.example` 复制为 `.env.local`，填写以下三项，重新启动开发服务并重载手机上的项目。基础地址包含服务商所需的版本前缀，应用会补上 `/chat/completions`。

| 环境变量 | 填写内容 |
| --- | --- |
| `EXPO_PUBLIC_LLM_BASE_URL` | 服务商的 HTTPS 基础地址 |
| `EXPO_PUBLIC_LLM_API_KEY` | 演示用密钥，仅在本地文件或个人构建环境填写 |
| `EXPO_PUBLIC_LLM_MODEL` | 该密钥可用的聊天模型标识 |

配置有效且填写题干时，触发和升级会通过内置 `fetch` 发送题干文字、触发类型、持续时间和所需层级。发送内容不含图片、笔迹坐标和会话历史。题干留空、配置缺失、网络或响应异常，或者请求超过 8 秒，提示卡会使用对应层级的本地文案并标注「当前为离线提示」。

Expo 会把 `EXPO_PUBLIC_*` 写入客户端程序，持有 APK 的人可以从中提取密钥。公开分发的 APK 建议省略这三项；现场在线演示可在受控设备中使用限额、可撤销的临时密钥。请将真实的 `.env.local`、签名文件与导出的笔迹留在本机。[Expo 环境变量说明](https://docs.expo.dev/guides/environment-variables/)

## 导出和查看 JSON

点「导出」可把当前会话保存到应用文档目录；弹窗中央显示可长按选择的完整文件 URI。点「另存到手机文件夹」，在安卓文件选择器中选择目录，之后即可用系统文件应用打开副本。「返回画布」可继续演算，建议在离开或重载前导出。

下面是一份表示结构的缩短示例。每个 `id` 标识一次操作或事件。`t` 和 `startedAt` 使用 Unix 整数毫秒，`x`、`y` 使用采集时窗口的逻辑坐标。

```json
{
  "meta": {"exportedAt": "2026-01-01T00:00:04.000Z", "strokeCount": 2, "pointCount": 3},
  "strokes": [
    {"id": 1, "points": [{"x": 44, "y": 96, "t": 1767225600000}, {"x": 58, "y": 96, "t": 1767225600017}]},
    {"id": 2, "tool": "eraser", "points": [{"x": 58, "y": 96, "t": 1767225600200}]}
  ],
  "stallEvents": []
}
```

`strokes` 保留尚未撤销的绘制操作，按顺序回放即可重现画面。橡皮带有 `tool: "eraser"`。撤销恢复上一步画面，已观测到的卡壳事件仍留在历史里。事件包含 `id`、`type`（`pause_erase`、`speed_drop` 或 `cognitive_stall`）、`startedAt`、`x`、`y` 和 `durationMs`。导出文件同时包含 `annotations`（伴随微标注标签如 `DEEP_THINKING_HOLD`、`SELF_CORRECTION`、`STUCK_CONCEPT_GAP`、`STUCK_CALCULATION`）与降采样 `stateTrace`（`[M, R, A]` 轨迹），标记为 `meta.version = "2.0-annotated"`，可用于 `ml/` 训练与消融验证。

## 端侧高级认知引擎与离线模型管线

- **题干图谱与大学数学示例库**：点「题干」可一键加载内置的 6 道带解题步骤树与歧路预警的大学数学例题（高数泰勒极限/变限积分/微分方程、线代含参方程组/实对称正交对角化、概率论全概率贝叶斯），断网环境即可立即测试毫秒级步骤匹配与歧路提醒。
- **端侧全科笔迹语法编码 (`EdgeInkSyntaxEncoder`)**：每笔落笔完成后在端侧执行几何与二维布局聚类，覆盖高数（`\int`、`\oint`、`\sum`、`\prod`、`\partial`、`\nabla`、`\infty`、`\to`）、线代（跨行矩阵/行列式界定符、上下标 `^T`/`_{ij}`）与概率统计（`~`、`\Phi`、`\cap`、`\cup`、`\varnothing`、`\bar{X}` 上划线），识别逻辑连接边界并计算轨迹对齐度 $A(t)$ 与边界惊奇倍率；预留 `IInkSyntaxEncoderAdapter` 接口便于挂载 MathWriting ONNX 模型。
- **零负担伴随微标注**：画布右上角悬浮轻量胶囊，支持一键标注「心算勿扰」（静默 8.5 秒防打扰）或主动选择「定理卡壳 / 算式受阻」，橡皮与撤销操作自动记入 `SELF_CORRECTION`。
- **离线模型训练与评测 (`ml/`)**：提供空间-动力学解耦双流 PyTorch 网络（`ml/models/decoupled_dual_stream.py`）、`v2.0-annotated` 数据转换器与 ONNX 导出脚本。运行 `python ml/train/benchmark_baseline.py` 可复现固定阈值与双流融合模型的消融实验对比。

## 如何打包成 APK

项目配置的安卓包名为 `app.inkpulse.notebook`，应用版本为 `1.1.0`，安卓版本号为 `2`。`eas.json` 中的 `preview` 配置指定内部安装与 APK 构建。

1. 在已安装依赖的仓库根目录安装 EAS CLI，使用自己的 Expo 账号登录并确认当前账号。

   ```sh
   npm install --global eas-cli
   eas login
   eas whoami
   ```

2. 首次构建执行 `eas init`，按提示创建或关联自己的 Expo 项目。CLI 会把该项目的标识写入 `app.json`。公开演示包可直接使用离线提示；仅限受控演示时，可在 Expo 项目的 **Environment variables → preview** 中设置前述三项，服务器无法读取本机被忽略的 `.env.local`。
3. 从仓库根目录提交安卓构建，首次按 EAS 提示创建或选择 Android 签名密钥。之后发布同包名的更新请沿用同一签名。

   ```sh
   eas build --platform android --profile preview
   ```

4. 打开终端给出的构建详情链接，等待状态变为 **Finished**，点下载获取 **`.apk` 安装文件**。也可以运行 `eas build:list --platform android` 找回记录。将下载链接发给测试设备，安装后直接点开「笔迹脉搏」，即可在开发电脑断开时使用本地提示。[Expo 首次构建指南](https://docs.expo.dev/build/setup/) · [Expo 安卓 APK 指南](https://docs.expo.dev/build-reference/apk/)

## 验收用例

| 验证内容 | 操作与预期结果 |
| --- | --- |
| 题干与在线提示 | 配置三项变量，输入上述等腰三角形题干并触发事件。提示应引用“等腰”“顶角”或“底角”等线索，L1 至 L3 逐步具体，均不给最终答案与完整解答。 |
| 离线及超时 | 清空密钥后重载，或在 APK 中断开网络，再触发卡壳。提示标注「当前为离线提示」，仍可点击升级；网络请求无响应时等待约 8 秒。 |
| 书写、橡皮与撤销 | 写两笔，用橡皮划过一部分后观察已擦区域，点「撤销」恢复这次擦除；再次撤销后最后一笔消失。 |
| 判定与冷却 | 分别按上文复现停顿涂改与速度骤降。冷却内重复动作，历史数不增加。另开空白会话，连续匀速移动 20 秒，确认无提示。 |
| 诊断与调参 | 连续划线查看近一秒点数，停笔后等待归零；打开详情分别移动三个滑块，核对显示值，用新阈值重试。 |
| 画像及导出 | 分别触发两类事件后核对「历史」计数，再另存 JSON，逐项比对事件类型、时刻和笔画；轻点、短横各一次应对应两条独立记录，点的时间戳递增。 |
| 屏幕适配 | 在手机竖屏、平板横竖屏打开题干键盘、诊断、提示、历史、导出，核对底部按钮可触及，较长内容可滚动，旋转后仍能继续书写。 |
| 安装与匿名 | 安装 preview APK，在无开发服务器时启动并跑一轮离线闭环；检查应用、README、注释与示例中只含演示题目、功能文案与匿名数据。 |

## 真机性能记录

早期采集版本在 OPD2506 平板（Android 16、Expo Go 58.0.5）上对每个完整一秒窗口逐一统计。120 Hz 自动连续划线 12.790 秒得到 115–122 点/秒；60 Hz 同动作 12.785 秒得到 59–62 点/秒；默认模式真人书写的两条长笔最低分别为 57、53 点/秒。这些数字描述触点送达速率；屏幕刷新率、实际渲染帧率和进程内存需要分别测量。

该平板报告的处理器型号为 MT8799，物理屏幕为 2120 × 3000，Android 内核可见总内存约 7.40 GiB（`MemTotal` 7,762,248 kB）。

在真机实测后填写下表。注明 Expo Go 或独立 APK，并在相同显示模式下连续演算 20 秒。可用 Android Studio 的 System Trace 或 Perfetto 查看实际呈现帧，用 `adb shell dumpsys meminfo app.inkpulse.notebook` 查看独立 APK 的 TOTAL PSS。Expo Go 应使用其自身包名，所测内存包含宿主进程。

| 版本 / 运行方式 | 设备 / Android | 处理器 | 物理内存 | 屏幕分辨率 / 刷新率 | 实际渲染帧率 | 书写后内存 PSS |
| --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |

## 代码与开源组件

- `src/capture/` 使用开源 Gesture Handler、Reanimated 与 Worklets 在 UI 线程接收触点，以单调时钟映射 Unix 毫秒。Skia 绘图随帧刷新，采样与显示分开执行。
- `src/stalls/` 的距离、速度滑窗、停顿与涂改匹配、冷却和事件分组由项目自研。界面使用 React Native 及开源 Safe Area Context 适配系统边缘。
- `src/engine/` 端侧多源认知引擎由 [团队名称] 全自研实现：包含几何语法编码与步骤对齐 (`inkSyntaxEncoder.ts`)、连续认知状态机 $[M, R, A]$ 与个人运动指纹 (`cognitiveState.ts`)、7 维 LinUCB 提示路由 (`personalizedPolicy.ts`)、解题图谱缓存 (`solutionCache.ts` & `sampleProblems.ts`)、题干多模态/文本摄入 (`problemIngestor.ts`)、端侧提示合成 (`edgeHintModel.ts`) 及伴随微标注采集 (`annotationTracker.ts`)。
- `src/hints/` 管理本地分层提示、防剧透校验与兼容服务请求。在线内容由实际配置的模型服务生成。
- `src/export/` 使用开源 expo-file-system 写入本机文件，先写临时文件再改名，安卓另存使用系统目录选择器。
- `ml/` 提供配套离线双流模型（PyTorch + ONNX 导出）与纯 CPU 基准评测流水线（基于 NumPy / scikit-learn）。

从 `main` 创建分支，并用中文写明设备、输入动作及验证结果。[贡献指南](CONTRIBUTING.md) · [MIT 许可证](LICENSE)
