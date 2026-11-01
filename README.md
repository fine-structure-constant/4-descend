# Connect4 BotLab Homework

四子棋 AI 的完整训练、搜索与平台发布记录。仓库保留 v1–v8 与 v10 的代码包、说明和评测记录；最新交付为 v10（本地验证，未确认上传）。

## 版本

| 标签 | 目录 | 类型 | 平台状态 |
|---|---|---|---|
| `v1` | `upload-v1/` | 小型双层 MLP，策略/价值头，基础 MCTS | 未确认 |
| `v2` | `upload-v2/` | 256 宽残差 MLP，专家/strong 数据与战术搜索 | 未确认 |
| `v3` | `upload-v3/` | 32 通道、4 残差块 CNN，残局搜索与开局参考数据 | number 1 · READY |
| `v4` | `upload-v4-exact12/` | v3 CNN + 完整 ≤12 子精确动作表，纯 Python | number 2 · READY |
| `v5` | `upload-v5/` | 精确监督 CNN + 逐列动作价值辅助头 | number 3 · READY |
| `v6` | `upload-v6/` | 窄残局/备用步求解、风险损失、972 条纠错 | number 4 · READY |
| `v7` | `upload-v7/` | 48 通道、41,661 条中盘精确记谱 | number 5 · READY |
| `v8` | `upload-v8/` | v7 权重 + Rust 离线精确求解常驻进程 | 未确认 |
| `v10` | `upload-v10/` | 纯 CNN（6 平面、192 通道、12 残差块）+ PUCT，18 轮自对弈 | 未确认 |

详细说明见 [VERSIONS.md](VERSIONS.md) 与 [docs/releases/](docs/releases/)。平台状态是历史快照，上传或参赛前需重新查询。

## 目录

| 路径 | 内容 |
|---|---|
| `upload-vN/` | 各版本发布包：`code.zip` 入库，`model.pt` 仅本地保留 |
| `docs/releases/` | 每个版本的统一说明（Markdown）与清单（JSON，含 SHA-256） |
| `experiments/` | 训练、搜索、求解器源码及各阶段评测记录 |
| `experiments/.checkpoint_store/` | 历史检查点归档库（本地保留，不入库） |
| `Homework-4-1030/` | 课程 starter、教程与作业说明 |
| `agent.md` | 平台接口、上传流程、版本映射与后续工作交接 |
| `VERSIONS.md` | v1–v8、v10 索引与产物摘要 |

## 仓库约定

- 纳入 git：源码、`code.zip`、训练/评测 JSON、日志、文档与报告。
- 不纳入 git：本地凭据、`model.pt` 等权重、检查点归档、公开数据集、缓存与原生构建产物。忽略规则见 [.gitignore](.gitignore)。
- 认证凭据只通过本地环境变量注入，仓库不保存任何 Cookie、会话或令牌。
- 每个版本的代码包与权重摘要记录在 `docs/releases/vN.json`；权重文件需在本地保留，或用摘要核对同一文件。
- 版本标签 `v1`–`v8` 与 `v10` 分别指向加入对应代码包的提交。

## 复现入口

课程环境使用 `pubpy`（CPU、单线程 PyTorch）。常用入口：

```bash
# 源码自检
cd experiments/exact12_v7
python check.py

# 导出当前搜索代码（会生成新目录，清单随之更新）
python export.py ../../upload-v7/model.pt --out ../../upload-v8-new

# 对官方模型评估（需本地认证环境）
python evaluate_api.py ../../upload-v8/model.pt --opponent strong \
  --games 8 --simulations 128 --output runs/strong_v8_new.json

# V10 纯 CNN 自检（CPU）
cd ../cnn_v10
python check.py
```

平台接口、上传步骤和认证注入方式见 [agent.md](agent.md)；v8 原生求解器的构建与验证见 [experiments/exact12_v7/README.md](experiments/exact12_v7/README.md)；v10 的 GPU 训练、导出与评测见 [experiments/cnn_v10/README.md](experiments/cnn_v10/README.md)。

## 文档

- [VERSIONS.md](VERSIONS.md)：版本索引与产物摘要
- [agent.md](agent.md)：平台接口、上传流程、版本映射与交接注意
- `experiments/exact12_v5/README.md`、`exact12_v6/README.md`、`exact12_v7/README.md`：v5–v8 训练与验证细节
- [experiments/cnn_v10/README.md](experiments/cnn_v10/README.md)：v10 纯 CNN 训练、导出与评测
- `experiments/v4_release/RELEASE.md`、`experiments/cnn_endgame_v4/ITERATION.md`：v3–v4 迭代记录
- [experiments/ARCHIVE_README.md](experiments/ARCHIVE_README.md)：检查点归档与恢复
- [experiments/cleanup_report.json](experiments/cleanup_report.json)：历史清理清单

## 许可

推理与训练代码包含 Pascal Pons 求解器的移植/参考实现。相关许可证与来源见各实验目录的 `REFERENCES.md`、`AGPL-3.0.txt` 与 `tools/AGPL-3.0.txt`。
