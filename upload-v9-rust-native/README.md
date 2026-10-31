# v9 Rust Python 扩展版 Baseline

平台搜索 **v9 纯求解器 Baseline**。最新为该 Bot 的平台版本 **4**，READY，
默认版本 ID：`414ab6a5-a6c3-45dd-a48c-74ed152469c7`。

- `code.zip`：上传的推理代码和 Rust 扩展，约 227 KB。
- `model.pt`：固定棋谱数据，89,769,902 字节，与上一版 v9 完全相同，没有学习参数。
- `source.zip`：完整 v9 Rust 扩展源码和构建方法，不重复附带棋谱。
- `release.json`：发布摘要、SHA-256 与平台测试结果。

按用户确认的助教允许 baseline 方式，编译后的 PyO3 扩展以 `.py` 条目打包，
由 `ExtensionFileLoader` 显式作为 ELF 库加载；该文件不是 Python 源码。
平台执红、执黄初始化和完整对局日志均确认 `rust-pyo3-abi3`、`native_ready=true`。
没有临时可执行文件、子进程、运行时编译、CNN 或 MCTS。

本地 27 个中盘冷启动均在 820 ms 预算内精确解出，最慢约 0.391 秒。
平台完整自对弈 42 手正常和棋，40 次自主落子全部与独立 C++ 求解器的
最优 WDL 一致；最高每手 CPU 142.55 ms、墙钟 165.89 ms。
这些是正确性和兼容性核验，不是跨对手胜率或全局完美求解保证。

先前 Python baseline 和发布包均保留。正式参赛指定未改动。
