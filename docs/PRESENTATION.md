# 音乐驱动的演出输出

原始研究权重生成的长线缺乏有效构图与音乐关联。本轮新增 `polish` 输出编排器，用已有音频缓存生成可回放、可跳转的表现层。它明确标记为 `audio_directed_presentation`，原始模型工程单独保留；**这属于程序化输出增强，没有重新训练模型，也不证明模型已经学会审美编舞**。

## 实际产物

- `artifacts/presentation/nwad.mp4`：对此前 5090 权重的 NWAD 原始工程应用输出增强，完整 159.438 秒。
- `artifacts/presentation/verruckt.mp4`：第二首测试歌曲的输出验证；来源是转换后的参考工程，不冒充第二首模型生成结果。
- 同目录 `.json` 是可编辑工程，`*-contact.png` 为不同时间静帧，`preview.png` 为浏览器实际预览。
- `preview/public/performance.json` 是默认演出示例。预览器仍支持旧版原始 SceneScore，关联对应音频后可试听。

## 表现规则

1. 频谱缓存中的低、中、高频能量分别驱动纵深、轮廓形变和细粒子。它们不代表可靠的鼓/人声/乐器分离。
2. 从 10 ms 特征的正向频谱变化检测瞬态，最小间隔约 160 ms；冲击从检测时间开始，以约 130 ms 衰减。火花和扩散环持续至多 650 ms，静音门控关闭动态元素。
3. 每 16 拍汇总能量，与歌曲内部的动态范围比较，平滑混合轮廓、波纹和纵深三种主构图。没有把任意检测的 16 拍片段宣称为准确的主歌/副歌标签。
4. 原始模型的不稳定坐标和缩放不进入本轮画面。`model_geometry_used=false` 写入工程；保留源工程哈希用于追踪。
5. 几何形状由绝对时间、缓存控制轨和固定种子计算，跳播不会改变结果；辉光只处理当前帧，不依赖播放历史。

五个图层可独立平移、缩放、旋转和调整透明度。预览器提供横向位置编辑，保存后仍可确定性回放。控制轨保存在 `metadata.presentation`：30 Hz 连续控制、精确瞬态时间、段落权重、种子与配色。这是 `performance-v1` 渲染扩展，第三方播放器需要实现此扩展；只有通用 Entity/Event 支持的播放器无法还原这些程序化几何。

## 使用

```powershell
.venv\Scripts\python.exe -m choreographer polish raw-score.json --audio song.ogg --features song.features.npz -o performance.json
.venv\Scripts\python.exe -m choreographer render performance.json --audio song.ogg --width 1280 --height 720 --fps 30 -o performance.mp4
```

`polish` 检查音频哈希、缓存与源工程时长，拒绝覆盖源工程。已有数据的演示生成命令：

```powershell
.venv\Scripts\python.exe -B scripts/presentation_demo.py --render
```

## 验证

- Python 14 项测试通过，新增覆盖静音、瞬态前不提前触发、衰减、确定性跳转、音频响应与错误音频拒绝。
- TypeScript 两项测试通过，包含与 Python 的控制值、几何点、颜色和透明度对照；不宣称 Canvas 与 Pillow 光晕逐像素一致。
- 新模式和旧模式均通过浏览器实际播放、跳转、编辑、移动布局和 WebM 录制检查。
- 已查看两首歌多个时间点的实际渲染静帧。观赏性仍由用户试听评判；没有人类偏好率或模型泛化结论。

## 参考

- [MilkDrop 官方介绍](https://www.geisswerks.com/milkdrop/)：波形空间与节拍触发特效。
- [Butterchurn 官方实现](https://github.com/jberg/butterchurn)：WebGL 音乐可视化与预设混合接口。
- [Synesthesia 音频控制文档](https://synesthesia.live/docs/ssf/audio_uniforms.html)：独立频段、瞬态与持续控制。

本实现采用上述设计思路，自行编写几何与渲染代码，未复制这些项目的预设、着色器或美术资产。
