# V10：纯 CNN 自对弈训练

V10 是一条独立的纯神经网络路线：推理时只有一个 6 个输入平面、192 通道、12 个残差块的 CNN，加上 PUCT 树搜索。代码包和权重中没有任何开局表、中盘表、残局求解器、原生二进制或联网调用；开局到残局全部由同一张网络评估。

本次从预热数据出发，完成 18 轮自对弈训练（合计 99,840 局、2,621,852 个局面），并用 v7 的历史棋谱预热、用其精确 WDL 标签做微调与持续监督。发布检查点为第 16 轮（`ckpt_round_0016.pt`），在 12,930 个留出精确局面上价值 MSE 0.1513、类别准确率 88.37%。

## 模型与搜索

- 输入 `6×6×7` 六个平面：己方棋子、对方棋子、空格、可落点、奇偶威胁行、行棋方。
- 网络：`Conv(6→192) + 12×ResBlock(192) + 策略头/价值头`，共 7,989,402 个参数。
- 价值头输出当前行棋方视角的胜率期望；策略头输出 7 列 logits。
- 训练搜索：批量 PUCT，`c_puct=1.5`，第一阶段自对弈 96 次模拟、第二阶段 128 次模拟，根节点 Dirichlet 噪声 `α=0.3, ε=0.25`；温度前 10 手 1.0、10–23 手 0.5、之后 0.15。
- 部署搜索：同样的 PUCT，64 次模拟，温度为 0（取访问数最大的合法列）。
- 水平镜像作为数据增强；棋盘规则、胜负和编码在 `game.py` 中用 42 位棋盘实现。

## 目录

| 文件 | 作用 |
|---|---|
| `game.py` | 规则、位棋盘、六平面编码、水平镜像 |
| `model.py` | 带 BatchNorm 的训练网络；导出时折叠 BatchNorm |
| `mcts.py` | 批量 PUCT、访问计数、Dirichlet 噪声 |
| `selfplay.py` | 单 GPU 多局并行的自对弈 worker，写 `.npz` 分片 |
| `train_ddp.py` | 10 卡 DDP、bf16、AdamW、余弦学习率、镜像增强 |
| `arena.py` | 两个检查点的成对换色本地竞技场，可指定开局序列 |
| `build_seed_data.py` | 从 v7 历史棋谱生成经规则复核的监督预热数据 |
| `build_exact_data.py` | 把 v7 精确 WDL/策略标签转成 V10 张量并划分留出集 |
| `evaluate_exact.py` | 在留出精确局面上测量价值 MSE、类别准确率与策略 top-1 |
| `init_checkpoint.py` | 生成随机初始化检查点 |
| `orchestrate.py` | 自对弈 → DDP 训练 → 竞技场的可续跑流水线 |
| `export.py` | 导出纯 Python 包、折叠 BN、验证大小和 CPU 单步耗时 |
| `evaluate_api.py` | 用官方 play 接口对 standard/strong 做换色评测 |
| `deploy/` | 平台 CPU 推理的 `DeployNet`、`Bot` 适配器 |
| `strategy.py` | 平台入口，导出 `Bot` |

## 训练数据与资源

训练回放、分片和中间检查点全部放在 `/tmp`，不进入用户 10 GiB 配额，也不入库：

```text
/tmp/cnn_v10_data/replays/round_XXXX/rank_XX.npz
/tmp/cnn_v10_data/seed/historical.npz
/tmp/cnn_v10_data/exact/exact_labels.npz
/tmp/cnn_v10_data/exact/exact_train.npz
/tmp/cnn_v10_runs/ckpt_round_XXXX.pt
/tmp/cnn_v10_runs/ckpt_exact_ft.pt
```

主机有 10 张 RTX 4080 16 GiB；`torchrun` 每卡一个自对弈进程，训练用 10 进程 DDP。每个 `.npz` 保存 `x`（uint8）、`policy`（float16 访问分布）、`value`（float16，当前行棋方视角）和 `ply`。小体积的轮次摘要、逐步训练日志、精确评测与竞技场结果已复制到 `runs/` 入库。

### 预热数据

`build_seed_data.py` 只接受能重放到终局且记录结果与棋盘一致的棋谱；本次从 `experiments/exact12_v7/data` 的 v7 自对弈、中盘和竞技场记录中得到 2,750 局、108,125 个局面。和棋占多数，因此决定性对局的样本默认重复 4 次；最终监督集 172,310 个局面。预热只用于初始化，推理仍然完全由 CNN 完成。

### 流水线

```bash
cd experiments/cnn_v10
python build_seed_data.py --data-dir ../exact12_v7/data \
  --out /tmp/cnn_v10_data/seed/historical.npz
python init_checkpoint.py --out /tmp/cnn_v10_runs/init.pt

python -m torch.distributed.run --standalone --nproc_per_node=10 train_ddp.py \
  --data '/tmp/cnn_v10_data/seed/historical.npz' \
  --out /tmp/cnn_v10_runs/pretrain.pt --epochs 6 --batch 128 --lr 0.002

python orchestrate.py --data /tmp/cnn_v10_data --runs /tmp/cnn_v10_runs \
  --init /tmp/cnn_v10_runs/pretrain.pt --rounds 12 \
  --games-per-worker 512 --concurrency 192 --sims 96 \
  --epochs 4 --batch 128 --lr 0.0015 --history-rounds 4 \
  --arena-games 64 --arena-sims 32 --devices 10 --max-hours 3
```

流水线每轮：10 卡自对弈生成回放 → 用最近 4 轮自对弈加预热数据做 4 个 epoch 的 DDP 训练 → 新旧检查点 64 局、32 模拟、成对换色竞技场。已完成的轮次会在重启时跳过。

### 精确微调与第二阶段

`build_exact_data.py` 从 v7 的 `refinement_v10.npz` 得到 129,038 个合法非终局局面（116,108 训练 / 12,930 留出），价值分布为 77,864 胜、27,826 和、23,348 负（均为当前行棋方视角）。第 12 轮后用精确训练集、预热数据和第 9–12 轮回放做 2 个 epoch、学习率 `3e-4` 的微调，共完成 808,805 个样本、1,262 步、46.1 秒；留出价值 MSE 从 0.3956 降到 0.1897。

第二阶段从微调检查点继续 6 轮（13–18）：每轮 10 卡生成 6,400 局、128 模拟，训练集加入精确训练集、预热数据和最近 4 轮回放，4 个 epoch、学习率 `1e-3`；竞技场改为 128 局、64 模拟、固定 8 手开局。留出指标在 13–16 轮继续改善，17–18 轮趋于平坦。

### 训练结果

留出精确集（12,930 个局面）上的价值与策略指标：

| 检查点 | 价值 MSE | 类别准确率 | 策略 top-1 最优 | 策略交叉熵 |
|---|---:|---:|---:|---:|
| 预热 | 0.7582 | 51.78% | 84.26% | 2.2018 |
| 第 12 轮 | 0.3956 | 73.02% | 91.41% | 2.0210 |
| 精确微调 | 0.1897 | 84.32% | 93.51% | 1.4590 |
| 第 13 轮 | 0.1605 | 86.91% | 94.80% | 1.4062 |
| 第 14 轮 | 0.1564 | 87.80% | 94.86% | 1.4219 |
| 第 15 轮 | 0.1543 | 87.87% | 94.65% | 1.4272 |
| **第 16 轮** | **0.1513** | **88.37%** | **94.82%** | 1.4413 |
| 第 17 轮 | 0.1517 | 88.46% | 94.60% | 1.4602 |
| 第 18 轮 | 0.1543 | 88.24% | 94.64% | 1.4683 |

本地竞技场（成对换色；除标注外为 64 模拟、固定 8 手开局）：

| 对局 | 局数 | 胜/和/负 | 得分 |
|---|---:|---|---:|
| 精确微调 vs 第 12 轮（固定开局） | 512 | 62 / 387 / 63 | 0.4990 |
| 精确微调 vs 第 12 轮（随机开局） | 512 | 199 / 113 / 200 | 0.4990 |
| 第 13 轮 vs 精确微调 | 128 | 3 / 125 / 0 | 0.5117 |
| 第 14 轮 vs 精确微调 | 128 | 3 / 125 / 0 | 0.5117 |
| 第 15 轮 vs 第 14 轮 | 128 | 0 / 128 / 0 | 0.5000 |
| 第 16 轮 vs 第 15 轮 | 128 | 0 / 125 / 3 | 0.4883 |
| 第 17 轮 vs 第 16 轮 | 128 | 0 / 125 / 3 | 0.4883 |
| 第 18 轮 vs 第 17 轮 | 128 | 3 / 125 / 0 | 0.5117 |
| 第 16 轮 vs 第 18 轮（128 模拟） | 128 | 0 / 128 / 0 | 0.5000 |
| 第 16 轮 vs 第 15 轮（随机开局） | 512 | 199 / 122 / 191 | 0.5078 |
| 第 16 轮 vs 第 15 轮（随机开局、128 模拟） | 128 | 55 / 25 / 48 | 0.5273 |

在固定 8 手开局下，第 16–18 轮之间几乎全部和棋（第 16 vs 第 18、128 模拟、128 局为 0/128/0）；随机开局能产生约 390 个胜负局，第 16 轮对第 15 轮净胜 8 局。发布检查点按留出精确指标选择第 16 轮。

## 导出与平台约束

```bash
python export.py /tmp/cnn_v10_runs/ckpt_round_0016.pt --out ../../upload-v10 --sims 64
```

导出脚本会：

- 把每个 BatchNorm 折叠进相邻卷积，生成仅含 `DeployNet` 的 `model.pt`；
- 校验折叠前后策略/价值误差小于 `1e-5`；
- 解压 `code.zip` 到临时目录，实际导入 `strategy.Bot` 并在多个中盘局面计时；
- 强制 `code.zip < 5,000,000` 字节、`model.pt < 100,000,000` 字节、单个真实局面的单线程 CPU 用时低于 `0.95 s`（可用 `--limit` 调整）。

本次发布（第 16 轮、64 模拟）：`code.zip` 5,008 字节，SHA-256 `9edf958d54f1bcc16416051444959174385acdabf04c9452a6aea891d4962c5b`；`model.pt` 31,954,402 字节，SHA-256 `d96b1f0afe1b41e4c6f73ad20b6452e8065776d5a10ed9ac630fcd1f9c0cc0c6`；BN 折叠最大误差 `3.34e-6`，探针最坏单线程 CPU 用时 0.728 秒。完整清单见 `runs/package_validation.json`。

## 自检

```bash
python check.py
```

覆盖随机对局的终局/胜负与 numpy 独立实现一致、棋盘往返、六平面镜像、访问计数总和等于模拟数、折叠网络输出一致。

## 评测

```bash
source ../../.botlab.env
python evaluate_api.py ../../upload-v10/model.pt --opponent standard \
  --games 40 --device cpu --output runs/api_standard_round16.json
python evaluate_api.py ../../upload-v10/model.pt --opponent strong \
  --games 40 --device cpu --output runs/api_strong_round16.json
```

官方 play 接口的换色评测（每组 40 局、64 模拟、CPU 单线程）：

| 检查点 | 对手 | 胜/和/负 | 得分 |
|---|---|---|---:|
| 第 15 轮（候选） | `standard` | 39 / 1 / 0 | 0.9875 |
| 第 15 轮（候选） | `strong` | 27 / 12 / 1 | 0.8250 |
| **第 16 轮（发布）** | `standard` | 36 / 2 / 2 | 0.9250 |
| **第 16 轮（发布）** | `strong` | 28 / 11 / 1 | 0.8375 |

40 局的抽样误差约 ±0.04，两个检查点的官方评测差异不显著。本地随机开局竞技场第 16 轮对第 15 轮为 512 局 0.5078、128 模拟 128 局 0.5273，发布检查点按留出精确指标（MSE 0.1513）选择第 16 轮。报告保存在 `runs/api_standard_round15.json`、`runs/api_strong_round15.json`、`runs/api_standard_round16.json`、`runs/api_strong_round16.json`。

`evaluate_api.py` 只读取官方 play 接口的应手，不创建平台比赛、不上传版本、不修改参赛设置；报告不包含 Cookie。评测使用导出的折叠网络和单线程 CPU，与平台推理路径一致。

## 边界

- 纯 CNN 的强度来自训练分布和搜索预算；它与 v7 的精确表/求解器不可直接类比。
- 官方 play 接口的 40 局评测使用固定开局序列与温度 0，和平台比赛的随机性、时限及对手分布不同，不能直接当作排名。
- 原始回放、分片和检查点在 `/tmp`，重启机器可能丢失；轮次摘要、逐步训练日志、精确评测与竞技场结果已复制到 `runs/`，最终 `upload-v10/` 与 `runs/package_validation.json` 保存在仓库目录。
- 未得到用户明确要求前，不上传平台、不创建付费比赛、不改变参赛版本。
