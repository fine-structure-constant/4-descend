v8：前 12 手藏谱 + 中盘表 + 嵌入式离线精确求解器

交付目录：homework4/upload-v8/code.zip、model.pt。v7 仍是正式比赛已指定的参赛版本；
v8 作为日常/对练版本上传，不改变参赛版本选择。平台版本 ID 与验证状态见下。

尺寸与摘要见 runs/release_manifest.json。code.zip 只含 .py/.md/.txt；model.pt 与 v7
相同（48 通道 CNN、12 手中盘表、9,188,816 条 ≤12 手最优走法表、14 手 Pons 开局表）。
v8 不重新训练网络，只扩展深层搜索。

相对 v7 的改动：

1. 嵌入式离线精确求解器。native/c4core 是 Pascal Pons AGPL 求解器的 Rust 端口
   （位棋盘、置换表、防直接送负、null-window 搜索、7x6.book 读取）。发布包把静态链接、
   strip 后的可执行文件压缩为文本放进 support/native_payload.py，运行时解码、校验
   SHA-256、解压到私有临时目录，以常驻子进程协议调用。首次 Bot 初始化时预热；
   进程随 Python 退出。AGPL 文本随包分发，对应源码在 native/c4core。

2. 触发范围严格限定。前 12 手仍走 movebook 藏谱路径；13–14 手仍走 Pons 开局表/
   中盘表；只有非训练（add_noise=False）、棋盘 ≥15 子、且中盘表未完整覆盖时才调用
   原生求解器。训练调用不会使用它。

3. 逐列精确 WDL 与回退。原生层返回每列胜/和/负/未知。完整解出时直接落子（相等结果
   用网络先验破同分）；部分解出时排除已证明更差的列，再交由原搜索；执行失败、超时
   或沙箱禁止执行时静默回退到 v7 的纯 Python 管线。时间上同时约束 CPU 与墙钟：
   原生预算最多 450 ms，总搜索墙钟上限 0.92 s，Python 侧 CPU 上限 0.80 s。

4. 常驻进程复用置换表。一次对局内只启动一次求解器，置换表跨回合保留；相比每次
   独立进程，省掉约 100 ms 启动/清零开销，并让后续回合复用已证明的上下界。

验证：

- 与带 7x6 开局表的 C++ 求解器对拍 24 个 14–24 子随机局面：24/24 解出，走法与
  每列 WDL 全部一致，0 错误（native/validate_vs_cpp.py）。
- 独立解压 code.zip 后加载 model.pt、构建 Bot、空盘藏谱落子与 16 子深层落子均通过；
  深层走法由 native_exact 阶段给出，用时约 0.04 s。
- 完整自对弈一局 27 手，全部合法，无超时；前 12 手未调用原生层。
- 原生层单次预算实测：50/200/450 ms 请求分别在 50.4/200.4/450.4 ms 返回。

构建与复现：

    # 工具链在本机 /tmp（重启后需重新安装）
    export RUSTUP_HOME=/tmp/rustup CARGO_HOME=/tmp/cargo PATH=/tmp/cargo/bin:$PATH
    cd native/c4core
    RUSTFLAGS="-C target-feature=+crt-static -C relocation-model=static" \
      cargo build --release --offline --target x86_64-unknown-linux-gnu
    cd ../..
    python native/make_payload.py
    python export.py ../../upload-v7/model.pt --out ../../upload-v8

关键文件：mcts.py 搜索协调；support/native.py 载荷解压/常驻客户端；support/bot.py
初始化预热与落子适配；support/native_payload.py 嵌入载荷；native/make_payload.py
重新生成载荷；native/validate_vs_cpp.py 对拍。公开算法与许可证见 REFERENCES.md、
AGPL-3.0.txt。
