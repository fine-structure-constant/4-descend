# v9 Baseline：延缓必败局面

本地已验证，尚未上传平台。源码：`experiments/solver_v9_pyo3/`。
原发布 `upload-v9-rust-native/` 保留。

确认必败后，统一使用 Rust 迭代加深搜索，尽量延后对手最快的强制获胜；
相同距离时优先减少对手安全应手、制造威胁。无法避免下一手输时，也尽量
封住更多直接获胜列。胜局和可和局面的原有证明与选步不变。

搜索只使用本手剩余预算，目标仍为 900 ms。日志仅在完成距离证明时填写
`resistance_plies`（半回合，含当前落子）；超时保留最后完成层的选择，
不宣称所有局面都能找到全局最长抵抗。延缓失败不等于保证对手会犯错。

独立核验：64 项有多个安全走法且抵抗距离不同的残局及镜像检查、6 条必败
出口、极短预算检查通过；旧比赛第 20 手固定局面从只能撑 2 个半回合改为
可撑 16 个半回合，经独立 C++ 完整分值复核。实际解包后的 32 个残局、
30 个 C++ 胜负对拍、4 个分析回退和 4 局完整自对弈通过，最高单手墙钟
约 880 ms。没有进行新平台实战，不能据此宣称胜率提升。

`code.zip` 236,609 字节；`model.pt` 89,769,902 字节，沿用原版 7 个固定
棋谱张量，无学习参数。`source.zip` 含修改后的完整搜索源码，不重复棋谱。
完整校验值和验证报告路径见 `release.json`。

扩展仍是 CPython abi3 >=3.8、Linux x86-64 的进程内 PyO3 模块，沿用原版
已获允许的 `.py` 条目包装方式；`support/_c4solver_native.py` 是 ELF，
须通过配套加载器加载，不能按普通 Python 源码编译。

```bash
python experiments/solver_v9_pyo3/validate_resistance.py --release upload-v9-rust-resistance
python experiments/solver_v9_pyo3/validate_package.py \
  --release upload-v9-rust-resistance \
  --output experiments/solver_v9_pyo3/runs/resistance_package_validation.json
```

独立 C++ 距离核验需要完整项目中的教师源码 `/tmp/mbwork` 和
`/tmp/7x6.book`。算法与许可证见源码包中的 REFERENCES.md、AGPL-3.0.txt
和 THIRD_PARTY_LICENSES.txt。
