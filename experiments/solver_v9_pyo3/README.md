# v9 Rust Python 扩展版 Baseline

纯棋谱 + Rust 精确求解器，无 CNN、MCTS 或学习参数。该修订独立于
`experiments/solver_v9/` 和 `upload-v9/`，原发布保留。

## 调用方式

本地 `support/_c4solver.abi3.so` 是 PyO3 构建的 CPython 稳定 ABI 扩展，兼容
Linux x86-64、CPython 3.8 及以上版本。平台按用户确认已获助教允许的 baseline
方式，把该 ELF 文件作为 `support/_c4solver_native.py` 条目打包，再通过
`ExtensionFileLoader` 显式作为扩展加载。它不是 Python 源码；不要直接 import
`_c4solver_native` 或对它运行 compileall。Python 直接调用内存中的 Rust
`Solver.analyze`，没有子进程、管道、临时可执行文件或运行时编译。
置换表约 64 MiB，跨回合保留。加载失败即初始化报错，不会冒充 Rust
版悄悄退回 Python。初始化日志包含 `native_backend=rust-pyo3-abi3`。

固定棋谱数据和上一版 v9 相同：完整 ≤12 子最优动作表、14 子分值表及
41,863 个精确中盘局面。`model.pt` 只有 7 个棋谱张量，PyTorch 仅用于读取。
初始化和背谱回合用于预算 13+ 子局面。主搜索没有深度截断，搜索到终局；
最多约 820 ms 的时间预算用尽后保留未知结论，因此不宣称所有局面都完美。

## 构建与运行

```bash
source /tmp/rust-env.sh
python native/build_extension.py
python export.py ../../upload-v9/model.pt --out ../../upload-v9-rust-native
python validate_package.py
python play.py --model ../../upload-v9-rust-native/model.pt --history 444444
```

本机共享 Cargo 缓存提供 PyO3 0.24.2，构建使用 `--offline`。
运行不需要 Rust 编译器或 PyO3 Python 包。只有构建时需要工具链。

平台上传 `code.zip` 和 `model.pt`。`source.zip` 提供本版本源码，不重复
附带棋谱，分享时三个文件一起保留。平台 READY 之后还需核验日志中的
实际 backend、托管完整对局、每手 CPU 以及独立求解器对拍。
正式参赛指定与 baseline 的默认版本分开管理。

## 核验

`validate_package.py` 对实际解压包执行独立数组残局穷举、C++ 教师对拍、
完整自对弈和无神经网络前向调用检查。更大规模的求解器性能和未知状态
核验记录于 `runs/`。胜负最优指 WDL，不保证最快获胜或最慢落败。

原先独立 Rust 程序在平台临时目录执行失败，不能推断 Python 扩展也不能
加载。此修订专门验证进程内扩展的兼容性。

算法、依赖和许可证见 `REFERENCES.md`、`AGPL-3.0.txt`、
`THIRD_PARTY_LICENSES.txt`。
