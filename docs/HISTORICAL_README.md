# Music Choreographer

研究版音乐条件几何编舞系统。输入整曲音频，输出可编辑的 `SceneScore`，由 Python 离线渲染器和 TypeScript / Three.js 预览器确定性回放。

**当前不是已经达到审美验收的生成产品。** 内置 Orbit 是明确标注的手工渲染示例；训练权重、真实谱面还原、手工演示分别记录来源，不混作模型效果。

本次真实训练、完整生成样本、失败案例与硬件实测见 [验证记录](docs/VALIDATION.md)。推理权重在 `runs/weights/choreographer-mert-pilot.pt`，实际模型视频在 `artifacts/mert-generated-full.mp4`。

新版 `choreography-v3` 已加入九种独立素材、十二种动作、DeepSeek 全曲规划和 11,120,640 参数的学生模型。当前预览使用实际偏好训练后的权重；[同曲对比](../preview/compare.html) 固定学生权重，仅更换计划来源。[实现与实验记录](docs/LLM_CHOREOGRAPHY.md) 说明人工选择、合成监督和实际未通过的检查。旧事件模型、动作模型、规则版与数学版继续保留。

```text
choreo analyze song.ogg --features song.npz -o analysis.json
choreo plan analysis.json --provider deepseek --style-brief "Smooth and natural transitions; preserve musical contrast" -o plan.json
choreo generate song.ogg --features song.npz --checkpoint artifacts/student-feedback/best.pt --planner model -o model.json
choreo generate song.ogg --features song.npz --checkpoint artifacts/student-feedback/best.pt --planner llm --plan plan.json -o llm.json
```

## 快速开始

Python 3.10+、FFmpeg、Node 20+。CUDA 使用 PyTorch 2.7 / CUDA 12.8；`torch` 与 `torchaudio` 必须同版本。

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.7.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -e ".[audio,test]"
cd preview
npm install
npm run dev
```

预览地址 `http://127.0.0.1:5173`。打开 SceneScore JSON，关联音频，可暂停、跳转、看轨迹、调整对象水平偏移并保存工程。WebM 为浏览器实时录制；需要固定帧率逐帧输出时用 `render`。所有输入文件均在本地处理。

## 五个接口

```bash
choreo convert chart.json --audio song.ogg --song-id song-family -o score.json
choreo train data/prepared/manifest.json --config configs/research.json --device cuda -o runs/research
choreo generate song.ogg --checkpoint runs/research/best.pt --device cuda -o generated.json
choreo render generated.json --audio song.ogg --width 1920 --height 1080 --fps 60 -o output.mp4
choreo evaluate generated.json -o evaluation.json --votes votes.csv
choreo generate song.ogg --baseline fixed --bpm 120 -o fixed-baseline.json
```

`generate` 默认提取与检查点匹配的音频特征；自动节拍检测失败会报错，可显式指定 `--bpm`。`--features` 可复用缓存，必须匹配音频 SHA256。`--condition zero|shuffle`、`--no-global` 用于消融；同 BPM 音频替换直接传入另一首音频并重新提取特征。固定种子用于复现实验，跨硬件/库版本的随机采样不保证逐位一致，保存的 SceneScore 可确定性播放。

`evaluate --pairs pairs.json` 生成随机 A/B 试次材料和单独的研究者答案文件；pairs 每项为 `song_id, full, local`，其中 full/local 是视频路径。给参与者的材料不应包含 `researcher_key.json`。教师强制音频条件诊断入口为 `scripts/condition_benchmark.py`。

公开谱面研究样本的获取与准备：

```bash
python scripts/fetch_pilot.py --songs 20 --output data/pilot
choreo prepare data/pilot/sources.json --lines-only --backend mert --device cuda -o data/prepared-mert
python scripts/fetch_phira.py --charts 20 --output data/phira
```

`--lines-only` 是**明确记录的判定线投影实验**，不含音符监督。去掉此选项将转换线和音符；不支持的效果写入 `quarantine.json`。默认 128 个实体词表，超出将隔离，不截断。现有官方谱面的音符总数常超过该限制，因此首轮采用线投影，尚未把完整演出谱数据集的全部视觉内容纳入训练。

`prepare` 保存谱面权威 BPM/offset，音频缓存另存 Beat This! 检测时间。`--backend spectral --bpm 120` 仅用于便宜的端到端基线与工程测试，不能作为 MERT / 自动节拍研究结果。若 Hugging Face 缓存下载卡住，可运行 `scripts/fetch_mert.py`，将 `CHOREO_MERT_PATH` 指向 `data/models/MERT-v1-95M`；该脚本固定上游 revision 并核对权重 SHA256。

## 原始事件模型与训练（保留兼容）

- 冻结 MERT，24 kHz，5 秒窗 / 1 秒重叠，75 Hz 表征，最后四层均值；按有效卷积帧中心时间合并。缓存不参与反向传播。
- 10 ms 频谱与正向谱通量进入可训练卷积，随后降采样，供局部交叉注意力使用。
- 全曲逐拍特征经过 4 层、256 维 Transformer，再汇总为 16 拍计划。计划含 8 维统计描述与训练集拟合的最多 64 个视觉代码。
- 8 层、512 维、8 头自回归解码器输出复合事件，完整配置 45,358,826 个可训练参数。小型 `pilot.json` 是 2,988,266 参数的 spectral 基线。
- 字段为操作、对象槽、属性、几何类型，以及拍间隔、持续时间、起终值、宽高。时间粗粒度为 1/192 拍，连续残差保留精度。每块绝对锚定，单属性采用后启动覆盖。
- 16 拍样本超过事件预算时递归细分并携带状态。相同歌曲家族及完全相同音频禁止跨划分。全曲视觉代码只用训练集拟合。
- 阶段 1 局部事件；阶段 2 加预测全曲计划和对象状态；阶段 3 加音乐/视觉对比损失，优先不同歌曲但相近 BPM/能量的微批次负样本。
- BF16、梯度检查点、SDPA、AdamW、梯度裁剪、余弦学习率、验证集早停和包含随机数状态的断点。跨阶段续训用 `--resume last.pt --stage 2 --epochs 40`；epochs 是累计上限。改变损失阶段时重置最佳验证损失比较。

计划始终由音频预测，不把真实计划喂给解码器。阶段 2/3 最后 25% 轮次，对 10% 样本用前块最多 64 步自回归预测的几何状态续写；保留真实生命周期以保证教师事件引用有效。这是有界的单块续写课程，尚不等同于整曲自由滚动训练。人体动作生成和三维审美训练不在本版范围。

## 共享服务器

本次专属目录：`/path/to/music-choreographer`。不修改其他目录。通过以下入口选择显存有余量的设备，始终排除物理 GPU 7 和已记录的 GPU 7 UUID：

```bash
PYTHONPATH=src .venv/bin/python scripts/shared_gpu.py --required-mb 8192 -- \
  .venv/bin/python -m choreographer train data/prepared/manifest.json \
  --config configs/pilot.json --device cuda --memory-limit-gb 30 -o runs/pilot
```

启动前查询显存，选择 GPU 7 以外空余最多的一张卡。`--required-mb 8192` 是启动所需的最低空余预算，不再表示固定的 8 GiB 限额。实际分配上限为 `min(30 GiB, 当前空闲显存 - 2 GiB)`，由环境变量传给训练、容量检查和服务器生成入口。一次只运行一个 GPU 任务；不停止其他任务。资源不足就失败退出，不自动改用 GPU 7。共享卡的空闲显存可能随后变化，这个上限不能保证其他任务启动后永不 OOM。

`scripts/server_validate.py` 是本次独立目录的验证入口：12 项测试、完整网络 BF16 检查、5 轮局部训练、3 轮规划训练、1 轮对齐训练和恢复后的第 10 轮训练。它要求 `.choreographer-owned` 标记，并拒绝覆盖已有的 `runs/server-5090`。`--epochs` 表示累计轮数，继续训练时应大于断点的已完成轮数。

长序列容量检查使用独立文件保存证据：

```bash
.venv/bin/python scripts/shared_gpu.py --required-mb 8192 -- \
  .venv/bin/python scripts/gpu_smoke.py --events 2048 --batch-size 2 \
  --output-prefix artifacts/gpu-capacity-2048
```

服务器新权重的单首未见整曲检查入口为 `scripts/server_generate.py`，也必须通过 `shared_gpu.py` 启动。实测结果与未验收项见 `docs/VALIDATION.md`。

## 音乐演出输出

原始模型工程可以通过新增的 `polish` 命令生成更清晰的表现层：频段分工、重音扩散环、火花、光晕及按乐句能量渐变的三种构图。

```bash
choreo polish raw-score.json --audio song.ogg --features song.features.npz -o performance.json
choreo render performance.json --audio song.ogg --width 1280 --height 720 --fps 30 -o performance.mp4
```

这是明确标注的规则编排，不代表新训练的编舞能力。原始模型输出保留以供比较。当前默认预览使用新版学生编排，“规则对照”保留此版本。实现、参考、使用方式和验证范围见 [docs/PRESENTATION.md](docs/PRESENTATION.md)。

## 从演出谱学习动作

新增 `actions` 分支将真实谱面的几何运动与音符时机作为监督，预测 64 类动作词、连续参数、持续可见性和强调时机。光环、丝带、纵深、火花作为执行器；推理不调用 `polish` 的音频规则。

这仍是**几何转译弱监督**：效果图元及“旋转对应光环”等动作语义由程序定义，谱师没有直接标注这些效果。新分支已在本地和服务器单卡训练，原始事件模型和规则版均保留。数据、训练记录、失败项及评估边界见 [docs/LEARNED_ACTIONS.md](docs/LEARNED_ACTIONS.md)。

```bash
choreo generate song.ogg --checkpoint runs/actions-direct/best.pt --features song.mert.npz -o learned.json
choreo actions evaluate --checkpoint runs/actions-direct/best.pt --manifest data/actions-v1/manifest.json -o artifacts/learned-actions-direct
```

打开生成的 `learned.json` 即可在预览器中播放与编辑。可点击“原始模型”“数学演出”“规则对照”查看历史版本；当前默认载入新版学生动作编排，来源均有标注。

服务器实验使用 GPU 0（RTX 5090），完成 15 轮 / 900 次更新并验证断点恢复。取回的权重是 `artifacts/actions-server/best.pt`，对应的 5 首完整场景、执行证据和两首样片在同一目录。空间编排尚未通过平均运动基线比较，当前结果按探索性实验记录。

## 数学动作库

`mathematical-v2` 将已有模型控制映射到玫瑰／利萨如／内旋轮线的闭合基底，用统一主图形、柔和惯性、有序粒子和细线抗锯齿改善动作执行。

```bash
choreo math learned.json -o mathematical.json
choreo render mathematical.json --audio song.ogg -o mathematical.mp4 --width 1280 --height 720 --fps 30
```

这次升级保留模型权重和预测重音时机；曲线基底与运动约束由程序设计，尚未训练新的数学图形选择策略。同模型前后对照、图元画廊、实际样片与验证边界见 [docs/MATHEMATICAL_ACTIONS.md](docs/MATHEMATICAL_ACTIONS.md)。

## 验证与限制

```bash
python -B -m pytest -q
cd preview && npm test && npm run build
```

覆盖变速、offset、父子旋转、曲线压缩、生命周期、跨块状态、同曲划分、粗量化残差、因果掩码、前反向/恢复、Python/TS 一致性。`scripts/browser_test.cjs` 验证播放、跳转、编辑下载、响应式布局和 WebM，需可用 Playwright 与 Edge。`scripts/gpu_smoke.py` 对完整网络跑 BF16 前向、反向和恢复，并保存显存报告。

内部轨迹审计是转换器与原生数学求值器的对照，**不是独立 Phigros/RPE 播放器逐像素验收**。样本几何皮肤、阴影、游戏判定反馈不宣称原样复现。自定义纹理、视频、着色器、非线性音符控制、负透明度特殊模式等会隔离。官方格式中跨流速变化的 hold 暂隔离。

盲评 CSV 字段：`song_id,full_preferred`，偏好完整模型记 1，局部模型记 0，平局记 0.5。先按歌曲平均，再按歌曲 bootstrap 95% CI；至少 20 首独立未见歌曲、均值 ≥60%、下界 >50% 才通过。首批总共 20 首、16/2/2 划分，不能达到此验收规模。最近邻描述符仅供排查，不能证明没有复刻轨迹。

## 参考与数据来源

- [Event-Based Token Sequences for Audio-Conditioned Music-Game Level Modeling](https://arxiv.org/html/2607.09095v1)：事件与相对拍间隔；本工程自行扩展空间、持久状态和全曲组织，并非论文官方实现。
- [MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M)：权重 CC-BY-NC-4.0；研究用途，不是商业授权。
- [Beat This!](https://github.com/CPJKU/beat_this)：自动节拍时间轴。
- [RPE 格式](https://docs.lchzh.top/learning/phiediter/)、[PhiZone Player](https://github.com/PhiZone/player)、[PhiChain](https://github.com/Ivan-1F/phichain)：格式语义核对，转换代码独立实现。
- [Phigros_Resource](https://github.com/7aGiven/Phigros_Resource)：公开镜像中的谱面与音频；下载器记录分支提交和文件 SHA256。公开可下载不等于获得再分发或商业训练许可，原始素材不放入代码发布包。
- [Phira](https://github.com/TeamFlos/phira)：通过公开谱面列表获取同人样本，记录谱面 ID、更新时间和包 SHA256；不登录，不猜测授权，不执行包内代码。

3D 可沿用时间与实体 ID，但需要新增三维变换、相机、材质/灯光字段和相应监督数据。当前 v1 坐标协议明确为 2D，不能把二维权重标为已具备三维编舞能力。
