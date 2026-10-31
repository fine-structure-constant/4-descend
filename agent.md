# Connect4 项目交接

记录日期：2026-10-31。工作根目录：`/home/stu2026/stu2400011444/homework4`。
第 1–9 节保留最初交接快照；后续已联网查询、上传和测试，最新结果见第 0 节。平台状态仍需按最新记录核实。

## 0. v9 纯求解器 Baseline（后续实测更新）

### 最新：Rust Python 扩展版已在平台成功运行

以下更新优先于后面的 Python 回退版快照。用户明确确认已问过助教，允许 baseline
将 Rust 编译扩展作为 `.py` 条目打包；无需再次询问该授权。

- 当前最新源码：`experiments/solver_v9_pyo3/`；发布包：`upload-v9-rust-native/`。旧 `solver_v9` 和 `upload-v9` 保留。
- 同一独立 Bot `67d71173-f14c-4e66-b971-ac85acd1fee9`，最新默认为平台 **number 4**，版本 ID `414ab6a5-a6c3-45dd-a48c-74ed152469c7`，READY / can_play。正式参赛接口未调用。
- PyO3 0.24.2、CPython abi3 >=3.8、Linux x86-64。扩展从提交目录直接加载，进程内调用 Rust，并保留约 64 MiB 置换表；无子进程、临时可执行文件或运行时编译。初始化强制要求扩展可用，失败即报错。
- 直接上传 `.so` 会收到 HTTP 400 INVALID_PACKAGE；实际通过的代码条目为 `support/_c4solver_native.py`，内容是 ELF 扩展，不是 Python 源码。`support/native.py` 通过 `ExtensionFileLoader("support._c4solver", path)` 显式加载，不能对该条目直接普通 import / compileall。
- **平台两方验证日志和整局日志均确认 native_backend=rust-pyo3-abi3、native_ready=true。** 先前“临时目录不能执行独立 Rust 程序”的结论不代表进程内 Python 扩展不可用。
- 平台自对弈 `65786554-4ec6-4451-84fa-6ea10984c69a`，随机预置 `[2,4]`（0-based），42 手正常和棋；40 次自主落子均与独立 C++ 教师最优 WDL 一致。每手最高 CPU **142.55 ms**，墙钟 **165.89 ms**。
- 本地实际解包通过 32 个独立残局、30 个 C++ 对拍和 4 局完整自对弈。另将 subprocess 禁用后测试 Rust 调用，27 个中盘冷启动全部在 820 ms 内解出，最慢约 0.391 s；总节点数冷态 18,887,145、热态 73,600。人工小样本不能当作全局完美证明。
- 数据仍为 7 个棋谱张量、零学习参数，41,863 条精确中盘局面。棋谱文件 SHA 与上一版 v9 相同。主求解器最多约 820 ms；未完成证明仍是未知。
- `code.zip` **226,809** 字节，SHA `98bed4329df38599081e974c8bd86f6ba0ac654eed762fb6290040193a9232ad`。
- `model.pt` **89,769,902** 字节，SHA `f258d43368bf7055577f927d79832f5f8e535d2946a3ecd908045cf060d1e345`。
- 构建：加载 `/tmp/rust-env.sh` 后，运行 `python native/build_extension.py`；发布：`python export.py ../../upload-v9/model.pt --out <新目录>`。
- 最终上传状态：`experiments/solver_v9_pyo3/runs/platform/pyo3_native_upload.json`；`pyo3_upload.json` 是直接 `.so` 被拒的会话，已取消释放预留空间。上传脚本支持 `--reuse-models` 并核对基础版本模型 SHA，可避免重复上传相同棋谱。
- 本次还产生 `upload-v9-rust/`（直接 `.so` 的本地包，平台拒绝）用于保留试验记录，不能误当成可上传的最终包。
- 证据：新源码 `runs/platform/native_validation_logs.json`、`native_smoke_logs.json`、`native_smoke_oracle_validation.json`、`runs/extension_benchmark.json`。分享说明和最终摘要在最新发布目录的 README.md / release.json。

### 上一发布：Python 回退版快照（平台 number 3）

本节覆盖下文关于“最新交付”“v8 未上传”的旧快照。2026-10-31 后续已实际上传和测试：

- 最新发布为 `upload-v9/`，源码独立放在 `experiments/solver_v9/`，不覆盖 v7/v8。
- v9 不创建 CNN，不使用 MCTS，学习参数为 0；`model.pt` 只有 7 个固定棋谱张量。完整 ≤12 子动作表、14 子分值表，以及扩展至 **41,863** 个局面的精确中盘表；新增 202 个离线证明局面是开发数据，不算独立棋力测试。
- 独立 Bot：**v9 纯求解器 Baseline**，ID `67d71173-f14c-4e66-b971-ac85acd1fee9`。
- 最新平台版本是该 Bot 的 **number 3**，ID `7b42e634-b80f-45d2-aeb9-93fe3e2458f6`，已 READY、can_play，已设为该 Bot 默认，供同学挑战；未调用正式参赛指定接口。number 1/2 是早期发布/诊断版，不应再当作最终 v9。
- **平台原生执行确实失败**：`/tmp`、`/dev/shm` 执行返回 EACCES，`/submission` 为只读。静态链接不解决执行权限限制。平台实际运行优化后的 Python 精确求解/有时限搜索；本地允许执行时可使用 Rust。证据在 `runs/platform/compatible_validation_logs.json`。
- 初始化及背谱回合都会预算 13+ 子局面，精确动作、置换表、已知胜负下界跨回合保留。Rust 单次最多约 820 ms；总决策目标 900 ms，背谱阶段预算最多约 800 ms。当前平台限制：每手 CPU 1 s / 墙钟 2 s；初始化 CPU 15 s / 墙钟 30 s。
- 主精确搜索没有固定截断深度，搜索到终局；时限未完成保留未知，必要时用纯启发式搜索。**不能宣称全局完美，也不能把可保和下界当成精确胜负值。**
- 本地正常路径和强制禁用原生路径，均通过 32 个独立残局穷举、30 个 C++ 对拍和 4 局完整自对弈。Python 路径最慢一手约 0.804 s。另有原生不同预算 81 次核验，以及禁用棋谱的 Python 内核上下界对拍。报告在 `experiments/solver_v9/runs/`。
- 最终兼容版平台自对弈 `8ca925ff-cf92-4498-9a96-ffa14d77b9f8` 正常结束，42 手和棋；最大 CPU 882.76 ms / 墙钟 904.08 ms。这是兼容性回归，不是独立棋力证明。分享说明和最终 ID 另见 `upload-v9/README.md`、`release.json`。
- 用户指出的败局 `d80ca7fb-9bc7-43e6-b11f-dadc34d6644e`：flash 执黑，v9 number 2 执白；随机预置 `[5,1]`（0-based）后黑方已必胜。后续 35 手独立 C++ 复核均保持最优 WDL，不能据此认定 v9 把可和局面下输了。报告 `runs/platform/flash_loss_oracle.json`。
- v8 曾上传 READY，ID `36082cc2-9689-4b59-8f84-6cf86d4b5699`；后续实时查询为 404，主 Bot 版本列表不含它，原因未查明。不要把历史 READY 当作当前仍可用。主 Bot v7 在本次列表快照中为 SELECTED。

上传/复现资料：`experiments/solver_v9/README.md`、`runs/release_manifest.json`。
最终上传状态文件为 `runs/platform/compatible_upload.json`，不要误用早期的 `final_upload_v9.json`。
分享 `upload-v9/code.zip`、`upload-v9/model.pt` 和 `upload-v9/source.zip`，源码包不含重复棋谱。

## 1. 接手先读

- 用户目标：代码包小于 **5,000,000 字节**，权重包小于 **100,000,000 字节**；前十二手精确记谱，之后结合网络与搜索，实战核验目标为稳定战胜 strong，并继续挑战 flash。旧版本的代码和参数需要留存。
- 最新交付是 `upload-v8/`，使用 v7 权重，新增离线原生精确求解器。当前开发源码仍在 **`experiments/exact12_v7/`**，这个目录已经演进成 v8，不能当作未经改动的 v7 源码。
- `upload-v7/` 是已上传且保存了 READY 响应的 v7。`upload-v8/` 已完成本地打包与验证，但本地没有找到 v8 上传提交/READY 记录，不能声称它已上传。
- `experiments/exact12_v7/runs/release_manifest.json` 现在指向 **v8**；v7 清单是同目录 `release_manifest_v7.json`。上传前必须确认使用哪份清单。
- **参赛版本未能确认**：当前 README 称 v7 已指定，但 `runs/flash_probe/platform_version.json` 的 v7 快照为 `competition_status: null`，`versions_before_upload.json` 的更早快照是 v6 为 SELECTED。先 GET `/competition` 核实，不能把 README 当成已执行指定的证据。日常上传与正式参赛指定是独立操作。
- 所有相对路径均相对于本文所在的工作根目录，除非命令先 `cd`。

## 2. 认证与请求约定

域名：`https://botlab.aiphys.cn`，API 前缀：`/api/v1`。

认证凭据只通过本地环境变量注入，不写入仓库、文档或代码包。脚本统一读取该环境变量；401/403 时先检查本地会话是否过期，不要用错误响应推断模型能力。凭据的获取与更新由使用者在仓库外完成。

请求约定：网页来源使用 `Origin: https://botlab.aiphys.cn`；`play` 使用 `/games/connect4/play` 作为 Referer，其余页面按对应页面路径。上传文件时 Content-Type 为带 boundary 的 multipart/form-data，其余 JSON 请求为 application/json。GET 是否强制认证取决于接口；现有脚本统一携带认证，未专门测试匿名权限。

通用认证与 HTTP 包装已实现于：

`experiments/exact12_v7/tools/botlab_peer.py` 的 `api(path, payload=None, method=None, content_type=..., key=None)`。

`path` 传 `/versions/...` 这种形式，函数自动添加域名和 `/api/v1`。有 payload 默认 POST，否则 GET。带 `key` 时设置 `Idempotency-Key`。脚本对 GET 或具幂等键的请求重试部分 502/503/504，仅允许跳转到本站并改用 HTTPS。

## 3. 上传平台的实际流程

发布目录只有 `code.zip` 和 `model.pt`。入口 `strategy.py` 导出 `from support.bot import Bot`，平台将单独上传的模型放在代码旁的 `models/model.pt`。Bot 使用 CPU、单线程 PyTorch；棋盘为 6×7，列号 0～6，黑先手为 1，白后手为 -1。

| 步骤 | 方法与路径（均加 `/api/v1`） | 请求内容/作用 |
|---|---|---|
| 查询自己版本 | GET `/bots/{bot_id}/versions` | 返回分页对象，版本在 `results` |
| 创建上传会话 | POST `/upload-sessions` | `bot, base_version, code_size, models_size, remove_models` |
| 上传代码 | PUT `/upload-sessions/{session_id}/files/code` | multipart 字段 `file`，文件 `code.zip` |
| 上传权重 | PUT `/upload-sessions/{session_id}/files/models` | multipart 字段 `file`，文件 `model.pt` |
| 提交版本 | POST `/upload-sessions/{session_id}/commit` | `note` 及 `receipts: {code: receipt_id, models: receipt_id}` |
| 等待验证 | GET `/versions/{version_id}` | 检查 `status == READY`、`can_play`、`validation_error` 及文件 SHA |
| 查询正式参赛 | GET `/competition` | 读取当前参赛/冻结状态 |
| 指定正式参赛 | POST `/competition/entry` | `{"version":"<version_id>"}`；**仅用户要求变更时执行** |

创建上传会话的样例：

```json
{
  "bot": "1e619e88-82a0-42d6-88f2-66199e68b402",
  "base_version": "65a6ca5f-a365-4f8d-a03a-31e02c33062e",
  "code_size": 852203,
  "models_size": 97503563,
  "remove_models": false
}
```

`base_version` 是沿用的基础版本，不是对手；上传大小必须取实际文件。commit 返回新版本 ID 后，需等待验证；提交成功不等于 READY。上传后同时核对服务端 `code.sha256`、`models.sha256` 与本地清单。

已有可恢复上传状态、保存 receipt、使用幂等键的脚本：

```bash
# 先在本机注入认证环境变量，仓库不保存任何凭据
python experiments/exact12_v7/tools/upload_v8.py --dry-run
# 确认确实要上传 v8 后：
python experiments/exact12_v7/tools/upload_v8.py \
  --base-version 65a6ca5f-a365-4f8d-a03a-31e02c33062e --wait
```

此脚本不指定参赛版本。默认状态文件是 `experiments/exact12_v7/runs/flash_probe/final_upload_v8.json`；若已有 commit，会直接返回原记录。新发布应指定新的 `--state`、`--tag`、`--note` 及对应 `--manifest`，不能复用旧版状态来假装上传新文件。`--dry-run` 也要求本地认证环境，且在存在历史版本记录时会发起只读查询。

旧 `botlab_peer.py upload` 是 v7 专用，有写死的 v7 幂等键和 note，现在默认清单却是 v8；**不要直接用它上传当前版本**。其 `OLD` 常量仍指向本地 v4-exact12，不代表最新版本。

平台曾以 HTTP 400 / INVALID_PACKAGE 拒绝包含 `.so` 和 C++ 文件的原生 v4 包。可接收包以 Python/文本文件为主，不能仅凭大小合格断言平台接收。v8 虽然 ZIP 条目都是文本，但 Python 载荷会解出原生可执行文件，能否在平台执行仍需实际验证；平台禁止时其运行逻辑会回退 v7 搜索。

## 4. 两类对战接口

### 4.1 本地模型下棋，官方 play 返回应手

```text
POST https://botlab.aiphys.cn/api/v1/games/connect4/play
```

```json
{"history":[4,3,3,3,3,4,3,4,2,4,4,2],"human_player":1,"model_id":"standard","action":1}
```

- `history` 是 action 之前的完整棋谱，0～6 列号；`action` 是本方新落子，别重复放进 history。
- `human_player` 实际代表客户端所控制的一方，可用本地模型代替人类，取 1 或 -1。后手空盘可不传 action，让官方先走。
- 官方标识已使用 `standard`、`strong`，是此接口的字符串 `model_id`，不能与平台版本 UUID 混淆。
- 返回中已使用并核验 `history, board, ai_action, terminal, winner`。
- 这是按提交历史重建局面的应手接口，客户端负责保存完整棋谱，可从指定历史开始分析。不能因此断言服务端不记录调用、不计额度或匿名可用；这些没有验证。
- 不要默认 `model_id="flash"` 可用。flash 已确认走下面的平台版本对战路径。

已实现自动对战、换先后手、独立规则校验、报告 SHA 与断点续跑：

```bash
cd experiments/exact12_v7
python evaluate_api.py ../../upload-v8/model.pt --opponent strong \
  --games 8 --simulations 128 --output runs/strong_v8_new.json
```

这里使用**当前目录源码**加载给定权重，因此上述测试是 v8 搜索+v7 权重；传 `upload-v7/model.pt` 也不会自动切回 v7 搜索。要测冻结 v7，先将 `upload-v7/code.zip` 独立解压，使用匹配的适配测试程序。

`evaluate_api.py` 支持 `--openings-file` 或 JSON 字符串 `--openings`；按同一开局交换先后手。`--resume` 会核验源码/权重 SHA，不能混合不同模型继续同一报告。遇到官方 HTTP 错误不可算成胜局。

### 4.2 平台托管双方模型，服务端运行整局

```text
POST /api/v1/evaluations
```

已成功使用的请求形式：

```json
{
  "mode": "peer",
  "version_a": "65a6ca5f-a365-4f8d-a03a-31e02c33062e",
  "version_b": "34b95b93-3588-4bd8-baf9-25d7170a8aef",
  "color": "white",
  "opening_pairs": 1
}
```

- `version_a` 是我方版本，`version_b` 是对手版本；`color` 指我方 black/white，黑先手、白后手。
- `opening_pairs=1` 表示预置双方各一步，共两步。实际初始棋谱要读响应/比赛记录里的 `opening_actions`，并计入完整回放。
- 返回含 `match`，随后服务端自行推进，客户端 GET 只负责观战，不推动落子。
- 身份由本地认证环境确定；JSON 不用重复传用户账号。上传包的推理不能依赖认证凭据或联网。
- **历史成功模式与前端快照有差异**：`/tmp/botlab-app.js` 当前下载副本使用 self/demo/compare 作为创建模式；现有 Python 脚本历史使用 peer。若新请求被参数校验拒绝，先核实当前前端/接口响应，不要盲目重试或认定所有模式都等价。

flash 版本 UUID：`34b95b93-3588-4bd8-baf9-25d7170a8aef`。这是历史已确认的对手版本，后续可再次核实其可用性。

```bash
python experiments/exact12_v7/tools/botlab_peer.py challenge \
  --version 65a6ca5f-a365-4f8d-a03a-31e02c33062e \
  --color white --opening-pairs 1 --label v7_flash_white_new1
```

每次新比赛使用新 `--label`。脚本会保存 `{label}_request.json` 的幂等键和 `{label}_creation.json`，已有 creation 时只接着查询原比赛。换先后手用另一个 label 和 `--color black`。两次独立创建会随机起步，不保证同一初始局面；严格成对统计需检查 `opening_actions`，不能仅凭 opening_pairs 相同就称同开局换色。

### 4.3 比赛状态、直播、余额与 active

| 接口 | 用途 |
|---|---|
| GET `/matches/{id}` | 比赛双方及版本、phase、step、winner、end_reason、opening_actions 等元数据 |
| GET `/matches/{id}/live` | 棋盘回放帧；已有响应含 frames，每帧有 step、board、drops、action、player |
| GET `/matches?active=1` | 筛选当前活动比赛；active 是查询参数，不是心跳写入 |
| GET `/evaluations/{id}` | 一次评测关联的比赛记录 |
| GET `/quotas` | 用户额度/余额。确切路径是复数 quotas |

拆成 detail/live 是将比赛信息与逐步棋盘帧分开提供，可按页面需求查询。前端终局判断可帮助停止刷新，但不能代替服务端的超时、崩溃、取消、判负和最终结果。

保存的前端代码中，活动比赛列表约每 2 秒刷新、对应 live 约每 1 秒刷新；quota 在不同页面有 10 秒或 30 秒刷新，不能当作后端要求的固定心跳频率。脚本 `wait_match` 约每 3 秒 GET detail，终态后只取一次 live；不需要为推进比赛持续 fetch，也不需要额外发送 active。

终态集合使用 COMPLETED/FAILED/CANCELED/REJECTED/ERROR。正常胜负需查看 `phase`、`winner` 和 `end_reason`；超时胜与正常棋力胜分开统计。

恢复观战：

```bash
python experiments/exact12_v7/tools/botlab_peer.py wait \
  --match beea5291-b20a-4992-a11d-427f8a793264 --label historical_match
```

这个历史 match 是用户最初提供的 v4-exact12 对 flash 实战。

## 5. 版本对应关系与类型

我方 Bot UUID：`1e619e88-82a0-42d6-88f2-66199e68b402`。**本地 vN 与平台 number 不一致**。

| 本地交付 | 主要类型 | 对应源码/记录 | 平台历史映射 |
|---|---|---|---|
| upload-v1 | 小型双层 MLP，策略/价值头，基础 MCTS | Homework-4-1030/第四次作业-Python脚本版/starter | 本次未确认 |
| upload-v2 | 256 宽残差 MLP，专家/strong 数据与战术搜索 | experiments/standard_v2 | 本次未确认 |
| upload-v3 | 32 通道、4 残差块 CNN，重力/奇偶特征，残局搜索及开局参考数据 | experiments/cnn_endgame_v4；standard_v3 是更早候选，不能直接等同交付 v3 | number 1，cfe7de9b-2193-4c69-9972-86307923f8c6，READY，note CNN 01 |
| upload-v4-exact12 | v3 CNN + 完整 ≤12 子精确动作表，纯 Python 推理 | experiments/cnn_endgame_v4 的 model_v4 导出；v4_release 内有相应测试记录 | number 2，95236ab0-16da-4220-89f8-6d4b4bd34daf，READY |
| 原生 v4 实验 | 另一路本地编译加速实现，不等于 v4-exact12 | experiments/v4_release/RELEASE.md | 含 .so/C++ 的包被拒；不能沿用该分支强度结果证明纯 Python 包强度 |
| upload-v5 | 精确监督 CNN + 逐列动作价值辅助头，精确数据重分析，增加 Pons 深开局参考 | experiments/exact12_v5 | number 3，0d3b3dae-1ccc-4b76-be09-1703efc24b3d，READY |
| upload-v6 | 32 通道，窄残局及顶端备用步完整求解、风险损失、972 条反事实纠错 | experiments/exact12_v6 | number 4，418cae24-b542-427a-ac9b-1572359f1ae8，READY |
| upload-v7 | 48 通道、195,517 参数，41,661 个逐列完整 WDL 中盘记谱局面 | experiments/exact12_v7；冻结源码以 upload-v7/code.zip 为准 | number 5，65a6ca5f-a365-4f8d-a03a-31e02c33062e，READY |
| upload-v8 | v7 原权重 + Rust 离线精确求解常驻进程，失败回退 Python | 当前 experiments/exact12_v7 | 只确认本地导出，尚无上传成功记录 |

完整 ≤12 子表有 **9,188,816** 个对称去重非终局局面；这里的手数为双方累计落子数。模型文件的大部分是非学习的棋谱缓冲区，不是近 100 MB 的神经网络。模型还含约 14 手 Pons 参考表。查表的最优动作集合/精确分值细节以 support/movebook.py、构建器与各版本记录为准，别混淆两个 v4 分支的覆盖口径和最短胜步口径。

冻结发布的大小及 SHA：

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| upload-v7/code.zip | 324286 | 9539999708ec298d32c304b270b17973ad4411843e3b5359405b51628b086656 |
| upload-v8/code.zip | 852203 | d5ff1abcfb19859d707fcb5ab59de1ca9605591094ad3521806cac1284482380 |
| upload-v7/model.pt 与 upload-v8/model.pt | 97503563 | 2a98dbb2d08d6c6c4b341e3d0746f3c8193541622e422c4ce05293a0228472ec |

## 6. 已有证据与强度边界

- 原生 v4 实验 strong 40 局 30 胜 9 平 1 负，但包被平台拒绝；**不能将此成绩归到已上传的纯 Python v4-exact12**。
- v4-exact12 对 flash：含用户提供局，共 1 胜 1 平 4 负，见 v4_release/runs/flash_probe/summary.json。
- v5：flash 8 局 3 胜 1 平 4 负；strong 8 局 4 胜 1 平 3 负。
- v6：strong 固定 20 组开局换色 40 局 23 胜 12 平 5 负，得分 72.5%，但未通过预设的全部配对稳定性条件。flash 平台结果 3 胜 3 平 2 负，其中 1 胜为对手超时；正常终局为 2 胜 3 平 2 负。
- v7：开发集 40 局对照 v6 权重为 19 胜 21 平 0 负，但双方共用 v7 搜索且复用开发开局；只能说明该开发对照，不等于两份完整发布版独立测试，也不等于稳定胜 flash/strong。见 runs/arena_v5nn2_fixed_vs_v6.json。
- v8：本地与 C++ 求解器对拍 24/24 一致，解包 Bot 测试与 27 手完整自对弈通过；单局最高墙钟约 0.803 秒。尚不能用这些证明平台兼容或对 flash 的实战强度。见 runs/package_validation_v8.json。

稳定性测试需冻结源码/权重、预先固定不同开局并换色、独立核验棋盘/终局，保存每手耗时；错误响应不计胜，超时另列。正式测试棋谱不能回灌同一轮候选后继续当作独立测试。

## 7. 训练、残局思路和下一步

用户关切：20～30 手剩三条主要可用列时，谁被迫垫棋决定胜负；其他列顶部仅一个空位也可能是关键节奏步。v6 起已处理：保留所有实际落点，只有经证明不可能参与任一方四连的单格才合并成节奏步，并保留剩余步数/先后手。超时返回未知，不能把启发值当证明。

精确标签训练：策略学习多个最优动作，价值学习精确胜负，动作价值辅助头学习逐列结果；风险项惩罚可胜变和/负、可和变负，全负局面不额外罚选列。未证明动作不当作平局。镜像等价局面整体划分训练/验证。自对弈后离线精确重分析，不仅用最终胜负粗标所有决策。

关键入口：`train.py`、`prepare_data.py`、`prepare_constrained.py`、`prepare_refutations.py`、`build_corrections.py`、`tools/build_midbook_labels.py`、`tools/pack_midbook.py`。v7 的 runs/*/config.json、best_metrics.json 和 log.jsonl 记录各候选配置。训练脚本默认模型/通道/输出仍偏旧版，续训 v7 必须显式传 48 通道和模型路径，先检查数据与 buffer 兼容，输出使用新目录。

v8 没有再训练网络。原生层只在非训练、至少 15 子且中盘表未完整覆盖时调用；前十二手不受影响。每列输出胜/和/负/未知，完整证明直接决策，部分证明只限制已证明更差的动作；CPU/墙钟约束和回退见 mcts.py。训练 add_noise=True 时禁用原生层。

v8 原生源代码：`experiments/exact12_v7/native/c4core`；载荷生成 `native/make_payload.py`；Python 客户端 `support/native.py`；文本载荷 `support/native_payload.py`。Rust 工具链在 `/tmp/rustup`、`/tmp/cargo`，临时目录丢失后需重装；现有载荷无需编译即可运行。许可证与公开算法来源见 REFERENCES.md、AGPL-3.0.txt。

导出新包时当前 export.py 会生成 v8 搜索包，并覆盖当前 release_manifest.json，输出目录必须不存在：

```bash
cd /home/stu2026/stu2400011444/homework4/experiments/exact12_v7
python export.py ../../upload-v7/model.pt --out ../../upload-v8-new
```

如继续迭代，优先核实当前参赛状态、v8 平台兼容性及真实换色战绩，再决定扩大网络或补充中盘精确标签。不要把训练指标、局部求解正确或开发集胜率直接写成稳定击败官方模型。

## 8. 磁盘、清理与恢复

历史清理将 70 份重复大检查点无损归档，experiments 从约 5.30 GB 降到约 620 MB。源码、训练数据、对局记录、最终上传包保留。`.pt.pack` 与 **`experiments/.checkpoint_store/` 必须一起保留**，不能把共享仓库当缓存删掉。

```bash
cd /home/stu2026/stu2400011444/homework4
python experiments/checkpoint_archive.py restore \
  experiments/exact12_v7/runs/v7_train2/model.pt.pack --out /tmp/restored_model.pt
```

完整说明见 experiments/ARCHIVE_README.md，清单见 experiments/cleanup_report.json。已交付 upload* 独立于归档仓库；v5/v6 最终模型及 v7 runs/release_v7/model.pt 仍有完整文件。恢复旧检查点时注意个人配额。

本会话此前的宿主机只读检查得到（容量为当时快照）：

| 路径 | 文件系统/限制 | 写权限与用途 |
|---|---|---|
| 用户目录 /home/stu2026/stu2400011444 | /dev/loop20，启用 usrquota；个人硬配额 **10 GiB**，当时已用约 3.76 GiB | 可写；整盘剩 180 GiB 不代表用户可用量 |
| /tmp、/var/tmp | 与系统根目录同在 /dev/nvme0n1p4；共享可用约 95.9 GiB；未启用上述用户盘配额 | 宿主机可写；两者容量不能相加，不是无限空间 |
| /dev/shm | tmpfs，上限约 125.8 GiB | 宿主机可写，使用内存，易失 |
| /data | /dev/sda1，剩约 41 TiB | 顶层及已检查的一级子目录无写权限；未递归检查全部子树 |
| /mnt/nfs_share | NFS | 全局 df 查询曾卡住，容量/可写性未确认 |

普通工具沙箱与宿主机权限不同：当前直接可写工作区与 /tmp；/var/tmp、/dev/shm 在普通沙箱可能只读。/tmp 的 tmpfiles 配置允许开机清理，适合可重建缓存，重要代码与权重不能只存这里。检查磁盘用按路径的 df，避免对无响应 NFS 做无界全盘查询。以上扫描未写入测试大文件、未迁移文件。

## 9. 交接注意

- 用户偏好直接推进已授权工作，少重复询问；但本次请求仅编写交接文档，没有要求再次上传/创建付费比赛/切换参赛版本。
- 文件名按用户要求是小写 `agent.md`，不是通常自动加载的 `AGENTS.md`。新会话请明确要求先读取本文。
- 作业完整源码提交与平台推理 code.zip 是两件事。课程说明要求服务器留完整训练/搜索代码，并让正式参赛版本与提交内容对应；只保留 code.zip 不能代替完整作业源码。
