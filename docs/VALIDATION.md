# 2026-10-13 实测记录

后续新增的音乐演出输出、两首实际视频和验证范围见 [PRESENTATION.md](PRESENTATION.md)。该输出层属于程序化增强，下面的原始模型质量结论仍然有效。

## 已完成

| 项目 | 结果 | 证据 |
|---|---|---|
| 官方谱面/音频 | 20 首，训练/验证/测试为 16/2/2 | `data/pilot/sources.json`、`data/prepared/manifest.json` |
| 判定线投影转换 | 20/20 通过内部轨迹审计，最大采样几何误差 0.172 px | `artifacts/data-audit.json` |
| MERT + Beat This! | 20/20 提取成功，含原音频哈希、编码器 revision、自动节拍 | `data/prepared-mert/manifest.json` |
| 完整网络 | 45,358,826 参数 | `configs/research.json`、`artifacts/gpu-smoke.json` |
| 本机完整网络检查 | RTX 3070 Ti Laptop，BF16 前向/反向/恢复通过 | `artifacts/gpu-smoke.json` |
| Python 自动化测试 | 12 项通过，包含完整训练中断恢复与生成可重复性 | `tests/` |
| 三阶段训练 | 局部 5 轮、规划/状态 3 轮、对齐 1 轮；428 个训练块、54 个验证块 | `runs/mert-local/metrics.jsonl` |
| 实际训练显存 | 本机峰值约 1.59 GiB；当前数据、微批次 2，不代表 2048 事件最坏情况 | `runs/mert-local/metrics.jsonl` |
| 未见整曲生成 | 159.438 秒、23 块、4 个实体、4964 个事件，2048 事件上限下全部块完成，结构校验通过 | `artifacts/mert-full-model-2048.json` |
| 同卡种子复现 | 完整整曲重新生成，4964 个事件逐项完全一致 | `artifacts/generation-reproducibility.json` |
| 视频 | 960×540、30 fps、H.264/AAC、159.438 秒 | `artifacts/mert-generated-full.mp4` |
| 预览器 | TS 构建、Python/TS 时间线一致性、播放/跳转/编辑下载/移动布局/WebM 通过 | `preview/src/runtime.test.ts`、`scripts/browser_test.cjs`、`artifacts/preview-*.png` |
| 共享服务器运行环境 | GPU 5 / RTX 5090 / torch 2.7.0+cu128，随机张量 BF16/反向/恢复成功，峰值 100.75 MiB | 远端 `runtime-smoke.json`；这是最初的基础环境测试 |
| 共享服务器完整模型 | GPU 6 / RTX 5090，45,358,826 参数，BF16 前向/反向/恢复通过，512 事件、微批次 1，峰值 1.222 GiB | 远端 `artifacts/gpu-smoke.json`；12 项 Python 测试也全部通过 |
| 5090 三阶段训练与续训 | 局部 5 轮、规划 3 轮、对齐 2 轮，共 140 个优化步骤；最后一轮从断点恢复；峰值 1.591 GiB | `artifacts/server-5090/metrics.jsonl`、`validation.json` |
| 2048 事件容量检查 | GPU 6，微批次 2，BF16 前向/反向/恢复通过；峰值分配 1.291 GiB，无 OOM；合成输入使用 1200 帧局部音频、512 拍全曲上下文 | `artifacts/server-5090/gpu-capacity-2048.json`；不代表所有音乐长度/BPM 的最坏情况 |
| 5090 新权重整曲生成 | NWAD 测试曲，159.438 秒、23 段全部结束、3 个实体、2615 个事件；约 31.9 秒生成，峰值分配 0.450 GiB | `artifacts/server-5090/generated-test.json`、`generation-evaluation.json` |

本次真实数据训练使用官方 EZ 判定线投影，没有把它包装成完整 RPE 演出谱监督。另从公开 Phira 列表获取了 16 个可明确选择谱面/音频的同人包，4 个包因多音频/格式等原因隔离。16 个包中，5 个判定线投影转换通过、11 个因自定义纹理/文本/时间映射等原因隔离。兼容性审计见 `data/phira-audit/`。例如 Kaleidoscope 的 26 条线、69072 个事件通过内部审计，采样最大误差约 0.192 px。这些新增同人包没有参与本次训练。

## 音乐条件诊断

教师强制事件损失，越低越好；每首歌按块取平均，统一排除计划/对比辅助损失。

| 测试曲 | 正确音频 | 清零音频 | 打乱时间 | 关闭全曲规划及状态 |
|---|---:|---:|---:|---:|
| NWAD | 9.6190 | 9.6270 | 9.6189 | 9.6562 |
| Verruckt | 7.6543 | 7.7053 | 7.6543 | 7.7309 |

证据：`artifacts/condition-benchmark-verified.json`。差异很小，打乱音频没有使损失明显变差。因此这批样本和训练轮次**没有证明模型学到了强音乐对应关系**。这些数值也不能替代自由生成、换歌测试或审美盲评。早期诊断曾漏载检查点，已作废并重新运行；作废文件带 `invalidated` 标记。

## 失败与未验收项

- 早期约 3M 参数 spectral 基线出现空输出、事件重复及过拟合。相关失败时间线保留在 `artifacts/model-pilot*.json`，不作为成功演示。
- MERT 模型在 512 事件限制下有 1 个片段未完成；按设计的 2048 上限重跑后该曲完成。只验证一首整曲的结构完整性，不能据此宣称所有歌曲可靠。
- 实际生成画面目前仍偏单线、运动少、颜色/构图不稳定，尚未达到“像人工演出谱一样有观赏性”的目标。
- 内部审计使用独立的原谱数学求值路径与 SceneScore 求值器，但尚无第三方播放器逐帧视觉对照。不能把“0.172 px”解释成完整游戏画面像素误差。
- 16 个同人包仍需严格兼容性筛选和人工审美筛选；并非已获得 20 首完整可用的优秀 RPE 演出监督。
- 仅 2 首测试集歌曲，没有 20 首未见歌曲盲评、60% 偏好率或显著性结论。
- 最多 128 个稳定实体 ID，暂不复用 ID；大量音符的完整谱面会被隔离。需要扩展动态对象槽与完整演出效果覆盖后再做全谱训练。
- 3D 数据、三维相机/材质/灯光解码和三维审美训练尚未实现；当前是版本化的 2D 协议。

## 交付物与运行边界

- 推理权重：`runs/weights/choreographer-mert-pilot.pt`（约 182 MB），以及 `local-baseline.pt`。仅作研究试验权重。
- 可继续训练的完整断点：`runs/mert-local/last.pt`。
- 5090 上的新训练断点及推理权重：远端专属目录的 `runs/server-5090/last.pt`、`inference.pt`。前 10 轮已完成，后续恢复时累计 `--epochs` 应大于 10。
- 实际生成工程/视频：`artifacts/mert-full-model-2048.json`、`artifacts/mert-generated-full.mp4`。
- `preview/public/demo.json` 和 `artifacts/demo.mp4` 是**手工演示**，仅验证播放器与编辑流程。
- 服务器仅新建 `/path/to/music-choreographer`，没有删除或修改原有项目目录，未使用 GPU 7。
- 用户明确批准后，1,037,511,215 字节研究包已上传并校验 SHA-256：`5994d4f90ded2839fac224705b2a11b2ed422b5ed7d6276aee0864a434b9679d`。包含项目代码、20 首官方谱面/音频及两类特征缓存，不包含新增 Phira 同人包、其他项目或凭证。验证脚本的后续代码更新单独同步。
- 服务器依赖安装在专属目录的 `.venv`；已完成的前 10 轮训练使用原 8 GiB 上限。用户随后授权使用一张 32 GB 卡，现改为单卡最高 30 GiB，按启动时空闲显存减去 2 GiB 余量动态下调。任务串行执行，始终排除物理 GPU 7 及其 UUID。没有停止其他用户任务。
- 新政策下本次容量与整曲生成任务实际预算为 11098 MiB（约 10.84 GiB），GPU 6 上其他任务仍在运行。`--required-mb 8192` 现在仅为启动最低预算；不再把本进程固定限制为 8 GiB。共享环境后续负载变化仍可能导致 OOM，不能宣称绝对保证。

后续研究的首要任务是补足可靠的演出谱监督与音乐条件学习，而不是扩大模型后直接宣称质量提升。
