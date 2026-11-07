# EmoD

EmoD 是一个**说话人音色—情感联合编码器**，与任何 TTS 主干解耦：输入参考音频，输出音色向量 `V_x` 与帧级情感向量 `V_emo(t)`；输入音素序列与情感标签，输出同一空间中的 `V_emo(t)`。消费方只需做一次加法 `V_x + V_emo`，即可得到目标向量。

本仓库是 `EmoD.tex` 所描述方案的工程实现，核心不是"再训一个情感分类器"，而是**在训练期把加性假设写成前向等式**：

```
z_t = V_x + V_emo,t          # additive bottleneck
mel_hat = Decoder(z_t, DPPG) # content only enters through DPPG
```

因此量化只出现在内容分支（DPPG），主表示始终是 VAE 连续潜变量——任何 VQ/FSQ 都会破坏 `q(a) + q(b) = q(a + b)`。

---

## 1. 目录结构

```
EmoD/
├── configs/
│   ├── _base_/            # 默认配置、前端、模型图
│   ├── data/esd.yaml      # ESD 说话人划分与目录映射
│   ├── phones/            # 中英音素表（40 EN + 56 ZH）
│   ├── gpu/               # 1/4/8 卡 3090、8 卡 H200
│   ├── stage1_ppg.yaml
│   ├── stage2_vae.yaml
│   └── stage3_t2e.yaml
├── emod/
│   ├── config.py          # _base_ 递归继承 + 相对路径解析
│   ├── data/              # 清单、ESD 适配器、分层采样、collate
│   ├── models/            # 前端、PPG/DPPG、加性 VAE、T2E
│   ├── trainers/          # accelerate 基座 + 三阶段
│   ├── losses.py          # 几何损失与课程权重
│   ├── calibration.py     # 方向表 / 尺度表标定
│   ├── inference.py       # 函数级向量接口
│   └── utils/             # 音频、检查点、日志、几何
├── scripts/               # 数据准备、对齐、启动、推理
├── tools/train.py         # 统一训练入口
├── docs/可行性分析.md
└── tests/                 # 配置/几何/DPPG/形状/检查点
```

## 2. 安装

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

CUDA 版本按机器选择，例如 3090 常用：

```bash
pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

## 3. 数据准备

### 3.1 ESD 原始目录

```
data/raw/ESD/0001/Angry/0001_000001.wav
data/raw/ESD/0001/Happy/0001_000002.wav
...
data/raw/ESD/0010/Surprise/0010_000350.wav
```

10 位说话人 × 350 条平行文本 × 5 类情感。**350 条文本在 10 位说话人之间共享**，因此同一 `text_id` 既可用于"同说话人跨情感差分"，也可用于"跨说话人同情感一致性"。

### 3.2 生成清单

```bash
python scripts/prepare_data.py --config configs/stage2_vae.yaml
```

默认 8 位训练、`0009` 验证、`0010` 测试（`configs/data/esd.yaml` 可改）。输出 `data/manifests/esd_{train,val,test}.csv`。

### 3.3 音素标注（阶段一、阶段三）

用 Montreal Forced Aligner 生成 TextGrid，放到与 wav 相同的相对路径下：

```
data/raw/ESD_TextGrid/0001/Angry/0001_000001.TextGrid
```

然后：

```bash
python scripts/make_phone_labels.py --config configs/stage1_ppg.yaml
```

得到 `data/phone_labels/<utt_id>.npy`（每 20 ms 一个音素 id）。若暂时没有对齐，可跳过阶段一与阶段三，先只跑阶段二。

## 4. 三阶段训练

三个阶段独立可运行，检查点格式统一（`format_version / stage / epoch / step / model / optimizer / scheduler / config / artifacts / metrics`）。

### 阶段一：内容分支（PPG → DPPG）

```bash
bash scripts/train_stage1.sh configs/stage1_ppg.yaml configs/gpu/rtx3090_4gpu.yaml
# 或
python scripts/launch.py --config configs/stage1_ppg.yaml --gpu configs/gpu/h200_8gpu.yaml
```

结束时在 rank 0 上按音素做 k-means，得到 DPPG 词表，写入 `checkpoints/stage1/final.pt` 的 `artifacts.dppg`。

### 阶段二：加性 VAE

```bash
bash scripts/train_stage2.sh configs/stage2_vae.yaml configs/gpu/rtx3090_4gpu.yaml
```

- 批内布局 `S × E × K`（默认 4×5×2=40/卡），保证同说话人跨情感、同情感跨说话人同时出现。
- 课程：0–5k 步只重构；5k 步后加属性分离；10k 步后加说话人对抗。
- 结束时自动标定方向表 `v̂_e`、尺度 `s_e`，写入 `checkpoints/stage2/final.pt` 的 `artifacts.directions`。

### 阶段三：T2E

```bash
bash scripts/train_stage3.sh configs/stage3_t2e.yaml configs/gpu/rtx3090_4gpu.yaml
```

教师为冻结的阶段二编码器；学生输出被约束在方向表上：`V_emo,t = α_t · s_e · v̂_e + r_t`，`‖r_t‖ ≤ 0.2 s_e`。

### 断点续训

```yaml
train:
  resume: auto        # auto -> <checkpoint_dir>/last.pt；none -> 从头；或写相对路径
```

所有阶段都支持；从 `last.pt` 恢复时按 `step % steps_per_epoch` 跳过已完成的批次。

## 5. 多卡与显存

| 配置 | 进程数 | 精度 | 每卡批 | 说明 |
|---|---|---|---|---|
| `gpu/rtx3090_1gpu.yaml` | 1 | fp16 | 8 / S=3 | 最小可跑 |
| `gpu/rtx3090_4gpu.yaml` | 4 | fp16 | 8 / S=4 | 推荐 |
| `gpu/rtx3090_8gpu.yaml` | 8 | fp16 | 8 / S=4 | 有效批 320 |
| `gpu/h200_8gpu.yaml` | 8 | bf16 | 24 / S=8 | 大 batch |
| `gpu/cpu_debug.yaml` | 1 | no | 2 | 仅 CPU 冒烟测试 |

阶段二的每卡批由采样器结构决定：`S × 情感数 × per_cell`（`per_cell` 默认 2）。例如 4 卡配置为 `4 × 5 × 2 = 40` 条/卡，8 卡 H200 配置为 `8 × 5 × 2 = 80` 条/卡。

`runtime.num_processes: auto` 解析为 `min(可见 GPU 数, 8)`。有效批大小固定，`grad_accum_steps` 自动吸收卡数差异：

```
accum = max(1, round(effective_batch_size / (per_gpu_batch_size * world_size)))
```

阶段二的每卡批由采样器结构决定（`num_speakers × num_emotions × per_cell`），不看 `per_gpu_batch_size`。

## 6. 推理接口

```python
from emod.inference import EmoDEncoder

enc = EmoDEncoder.from_checkpoint("checkpoints/stage2/final.pt", device="cuda")
ref = enc.encode_reference("ref.wav")

# 1) 用情感标签：V_emo(tar) = λ · s_e · v̂_e
target = enc.target_from_label("happy", ref.emotion_frames.shape[0], intensity=1.2)
joint = enc.compose(ref, target)          # [T, 64]，即 V_emo(tar),x

# 2) 用另一段音频的情感（EmoRES / EmoSteer 式替换）
joint = enc.compose_from_audio(ref, "target.wav")

# 3) 接入主干：按主干帧率重采样
frames = enc.to_rate(joint, 93.75)        # F5-TTS: 24 kHz / hop 256
```

T2E：

```python
from emod.inference import EmoDT2E

t2e = EmoDT2E.from_checkpoint("checkpoints/stage3/final.pt", device="cuda")
out = t2e.from_phones(phone_ids, "angry", durations=None)   # durations=None -> 时长预测器
```

`from_text` 需要外部 G2P，本仓库不内置。

## 7. 配置

- 所有超参在 YAML，代码内无硬编码路径。
- `_base_` 递归继承，后者覆盖前者；列表整体替换，字典递归合并。
- CLI 覆盖：`--set train.lr=1e-5 data.num_workers=2`。
- 所有路径相对项目根目录，通过 `emod.config.resolve_path` 解析。

## 8. 自检

```bash
python -m compileall emod tools scripts
pytest -q                                  # 若第三方 pytest 插件报错，加 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python tools/train.py --config configs/stage2_vae.yaml --dry-run   # 需数据与阶段一检查点
```

## 9. 已知限制

1. **缺少强度标注**：强度由投影系数给出，训练不依赖强度标签；评测需 CREMA-D / MEAD。
2. **DPPG 依赖强制对齐**：没有 TextGrid 时，阶段二会退化为"音素 id + 0 号变体"，仍可运行但内容条件变弱。
3. **ESD 只有 10 位说话人**：说话人分类头容易过拟合，建议后续用合成反事实语料扩充（见知识库 `相关数据集说明.md`）。
4. **教师特征未缓存**：阶段三每步都跑一次冻结前端，是当前主要瓶颈；缓存后可显著加速。
