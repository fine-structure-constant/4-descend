# V10：纯 CNN 自对弈训练

V10 是一条独立的纯神经网络路线：推理时只有一个 6 层输入平面、192 通道、12 个残差块的 CNN，加上 PUCT 树搜索。代码包和权重中没有任何开局表、中盘表、残局求解器、原生二进制或联网调用；开局到残局全部由同一张网络评估。

## 模型与搜索

- 输入 `6×6×7` 六个平面：己方棋子、对方棋子、空格、可落点、奇偶威胁行、行棋方。
- 网络：`Conv(6→192) + 12×ResBlock(192) + 策略头/价值头`，共 7,989,402 个参数。
- 价值头输出当前行棋方视角的胜率期望；策略头输出 7 列 logits。
- 训练搜索：批量 PUCT，`c_puct=1.5`，自对弈 96 次模拟，根节点 Dirichlet 噪声 `α=0.3, ε=0.25`；温度前 10 手 1.0、10–23 手 0.5、之后 0.15。
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
| `arena.py` | 两个检查点的成对换色本地竞技场 |
| `build_seed_data.py` | 从 v7 历史棋谱生成经规则复核的监督预热数据 |
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
/tmp/cnn_v10_runs/ckpt_round_XXXX.pt
```

主机有 10 张 RTX 4080 16 GiB；`torchrun` 每卡一个自对弈进程，训练用 10 进程 DDP。每个 `.npz` 保存 `x`（uint8）、`policy`（float16 访问分布）、`value`（float16，当前行棋方视角）和 `ply`。

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

## 导出与平台约束

```bash
python export.py /tmp/cnn_v10_runs/ckpt_round_0012.pt --out ../../upload-v10 --sims 64
```

导出脚本会：

- 把每个 BatchNorm 折叠进相邻卷积，生成仅含 `DeployNet` 的 `model.pt`；
- 校验折叠前后策略/价值误差小于 `1e-5`；
- 解压 `code.zip` 到临时目录，实际导入 `strategy.Bot` 并在多个中盘局面计时；
- 强制 `code.zip < 5,000,000` 字节、`model.pt < 100,000,000` 字节、单个真实局面的单线程 CPU 用时低于 `0.95 s`（可用 `--limit` 调整）。

本次导出：`code.zip` 约 5 KB，`model.pt` 约 32 MB；64 模拟的最坏单步 CPU 时间见 `runs/package_validation.json`。

## 自检

```bash
python check.py
```

覆盖随机对局的终局/胜负与 numpy 独立实现一致、棋盘往返、六平面镜像、访问计数总和等于模拟数、折叠网络输出一致。

## 评测

```bash
source ../../.botlab.env
python evaluate_api.py ../../upload-v10/model.pt --opponent strong \
  --games 8 --device cpu --output runs/strong_v10.json
```

`evaluate_api.py` 只读取官方 play 接口的应手，不创建平台比赛、不上传版本、不修改参赛设置；报告不包含 Cookie。评测使用导出的折叠网络和单线程 CPU，与平台推理路径一致。

## 边界

- 纯 CNN 的强度来自训练分布和搜索预算；它与 v7 的精确表/求解器不可直接类比。
- 自对弈数据、检查点和训练日志在 `/tmp`，重启机器可能丢失；最终 `upload-v10/` 与 `runs/package_validation.json` 保存在仓库目录。
- 未得到用户明确要求前，不上传平台、不创建付费比赛、不改变参赛版本。
