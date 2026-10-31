# v9 纯求解器 Baseline

固定精确棋谱 + 静态链接 Rust 终局求解器。没有 CNN、神经网络前向计算或
MCTS，学习参数数量为 0。`model.pt` 仅保存棋谱张量，PyTorch 仅用于读取数据。

## 使用

平台上传 `upload-v9/code.zip` 和 `upload-v9/model.pt`，无需解压。
独立的 v9 Baseline Bot 设置默认版本后，同学可在“同学对战”中按名称搜索。
具体平台 Bot/版本编号记录在 `runs/platform/`。

分享文件：`code.zip`、`model.pt`、`source.zip`。前两项直接用于平台上传，
第三项提供完整的本版本源代码。源码包不包含约 90 MB 的重复棋谱；请保留
并单独传递 `model.pt`。

本地询问一个局面的落子（列号从 1 到 7）：

```bash
cd experiments/solver_v9
python play.py --model ../../upload-v9/model.pt --history 444444
```

## 决策过程

1. 棋盘已有不超过 12 子时，查完整精确最优动作表，确定当前落子。
2. 13 子时可利用 14 子精确分值表评估下一步；中盘精确表也直接提供证明。
3. 未覆盖局面由 Rust 搜索到终局，使用位棋盘、直接败着剪枝、α–β、
   零窗口搜索、威胁排序及约 64 MiB 常驻置换表。根节点轮流分配时间，
   防止某一难解走法独占全部预算；上下界吻合后合并为精确缓存。
4. 已证明胜着优先；不能证明胜利时优先已证明可和的动作。仍未解决时，
   使用有时限的纯搜索及人工启发式，不把估值报告为精确结果。
5. 禁止执行原生程序的主机，回退到 Python 精确搜索及有时限的纯搜索。
   本平台实测原生启动失败，因此平台版必须按 Python 回退路径核验，不能
   将本地 Rust 求解速度直接作为平台速度。初始化日志会报告实际引擎。

中盘表共 41,863 个局面，包含在原 v7 表基础上离线证明的 202 个新增局面
和替代应手。它们是开发数据，不能用同一批局面统计独立棋力提升。

每手主求解器最多约 820 ms，总决策目标为 900 ms。当前平台限制为
1 秒 CPU、2 秒墙钟，初始化为 15 秒 CPU、30 秒墙钟。

初始化时预算中央开局。背谱已确定当前动作后，利用约 800 ms 以内的空余
时间，沿本方背谱选择、枚举对手可能应手，从已有 13 子的局面起提前求解。
原生不可用时同样利用这些时间执行 Python 预算。已知胜负下界跨回合传递，
帮助排除不可能保住原胜负值的走法。
精确动作和搜索置换表跨回合保留；没有后台线程，也不会在等待对手时偷跑。
平台等待对手期间会暂停进程。

有限时间下仍可能有未解决局面；本版本不宣称所有可达局面均完美求解。
`last_info` 区分精确证明、预算命中和未完成搜索。胜负最优指 WDL 最优，
不保证所有中盘选择都是最快取胜或最慢落败。

## 构建与核验

当前工作机共享 Rust 工具链在 `/tmp`，可先加载 `/tmp/rust-env.sh`：

```bash
source /tmp/rust-env.sh
RUSTFLAGS='-C target-feature=+crt-static -C relocation-model=static' \
  cargo test --offline --manifest-path native/c4core/Cargo.toml --target x86_64-unknown-linux-gnu
RUSTFLAGS='-C target-feature=+crt-static -C relocation-model=static' \
  cargo build --release --offline --manifest-path native/c4core/Cargo.toml --target x86_64-unknown-linux-gnu
python native/make_payload.py
python export.py ../../upload-v9/model.pt --out ../../upload-v9-rebuilt
```

平台推理不需要编译器或网络。已有静态载荷可直接使用。
原生构建目标为 Linux x86-64，不要求本机 CPU 特有指令。

完整项目中运行 `python validate_package.py`，核验解包后的实际 Bot：
独立数组棋盘穷举残局、C++ 求解器对拍、执行受限回退和完整自对弈。
报告位于 `runs/package_validation.json`。C++ 对拍依赖完整项目的教师程序
与公开 `7x6.book`；单独源码包不重复附带它们。

算法和数据来源见 `REFERENCES.md`，许可证见 `AGPL-3.0.txt`。
