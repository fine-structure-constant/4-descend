# v9 Rust 对 v7 / v6：20 盘平台测试

日期：2026-11-01（Asia/Shanghai）。固定已发布版本，测试过程中未更改代码、棋谱或参数。

v9：`414ab6a5-a6c3-45dd-a48c-74ed152469c7`；v7：`65a6ca5f-a365-4f8d-a03a-31e02c33062e`；v6：`418cae24-b542-427a-ac9b-1572359f1ae8`。

总计 20 盘，各对手 10 盘，v9 对各对手执黑 5 盘、执白 5 盘。每盘设置 `opening_pairs=1`：双方各预置一手、左右镜像。开局由平台独立随机产生，没有控制相同种子的严格同开局换色，因此各组开局优劣分布不相同。

| 对手 | 胜 | 和 | 负 | 胜率 | 得分率 |
|---|---:|---:|---:|---:|---:|
| v7 | 3 | 3 | 4 | 30% | 45% |
| v6 | 2 | 5 | 3 | 20% | 45% |

得分率 =（胜 + 0.5 × 和）/ 盘数。20 盘均正常终局，无超时、崩溃或非法落子。

## 开局与独立核验

独立 C++ 精确求解器复核 v9 的 **290 次自主落子**，全部保持当时局面的最优胜负值；无未完成的对拍项。该核验针对本批实际走过的局面，不是全局完美求解证明。

| 对手 | v9 开局必胜 | 开局理论和棋 | v9 开局必败 |
|---|---:|---:|---:|
| v7 | 3 | 3 | 4 |
| v6 | 1 | 6 | 3 |

全部 7 场负局都在预置随机开局之后已理论必败；全部开局必胜局均兑现，理论和棋局没有输。对 v6 有 1 局从理论和棋转为胜利。

对 v6 的该局：`6e1d7a7b-cd97-465d-88b8-4d884c59b468`，v9 执白。v6 第 15 手选择第 5 列，把可和局面变为必败；独立求解器给出的唯一保和列是第 4 列（均为 1-based）。

就这批具体开局，若双方一直保持最优 WDL，v9 对 v7 的预期得分率为 45%，对 v6 为 40%；实际分别为 45%、45%。因此不能仅凭原始胜率较低判断 v9 出现了棋力问题。

v9 全部对局最高单手 CPU **800.59 ms**、墙钟 **823.04 ms**；每盘运行日志均确认 `rust-pyo3-abi3` 与 `native_ready=true`。

## 逐局结果

| 序号 | 对手 | v9 先后手 | 预置开局（1-based） | v9 初始 WDL | 结果 | 平台回放 |
|---|---|---|---|---:|---|---|
| 1 | v7 | 先手 | 6,2 | 1 | 胜 | [回放](https://botlab.aiphys.cn/matches/4efbfa7f-1830-4790-950a-a7d122bc2307) |
| 2 | v7 | 后手 | 7,1 | -1 | 负 | [回放](https://botlab.aiphys.cn/matches/60bb9eba-1d10-4457-bd3f-1c229233e29f) |
| 3 | v6 | 先手 | 2,6 | 1 | 胜 | [回放](https://botlab.aiphys.cn/matches/de8907d2-3db8-4b98-b239-6a2e1b049a63) |
| 4 | v6 | 后手 | 2,6 | -1 | 负 | [回放](https://botlab.aiphys.cn/matches/8a351e13-8528-4a34-a272-8569458ea9b2) |
| 5 | v7 | 先手 | 2,6 | 1 | 胜 | [回放](https://botlab.aiphys.cn/matches/eccb5734-6df0-4ade-af81-960d58748566) |
| 6 | v7 | 后手 | 2,6 | -1 | 负 | [回放](https://botlab.aiphys.cn/matches/c6d387f5-e9bc-4231-86ee-eaf35a9a4fd2) |
| 7 | v6 | 先手 | 3,5 | 0 | 和 | [回放](https://botlab.aiphys.cn/matches/e5bda6fb-53fc-4fc8-9c1a-d27139710c2c) |
| 8 | v6 | 后手 | 2,6 | -1 | 负 | [回放](https://botlab.aiphys.cn/matches/12a5077d-6d8a-4431-9ec1-06f4a625a3cb) |
| 9 | v7 | 先手 | 3,5 | 0 | 和 | [回放](https://botlab.aiphys.cn/matches/e1bdda95-8cf0-44a2-b147-a42d91bf1cc0) |
| 10 | v7 | 后手 | 5,3 | 0 | 和 | [回放](https://botlab.aiphys.cn/matches/c65e38cf-093f-4f02-97ff-5553e003acb8) |
| 11 | v6 | 先手 | 3,5 | 0 | 和 | [回放](https://botlab.aiphys.cn/matches/acf084f9-2dcc-422a-b2ab-e87453be7cf0) |
| 12 | v6 | 后手 | 5,3 | 0 | 和 | [回放](https://botlab.aiphys.cn/matches/1a9e3a76-2493-4daa-b731-8066191454b2) |
| 13 | v7 | 先手 | 2,6 | 1 | 胜 | [回放](https://botlab.aiphys.cn/matches/0ff7f436-27c8-4dd9-83f6-25b0942f6c60) |
| 14 | v7 | 后手 | 7,1 | -1 | 负 | [回放](https://botlab.aiphys.cn/matches/f0ecfed8-ddb0-4ddb-9923-abc03b11a2a4) |
| 15 | v6 | 先手 | 5,3 | 0 | 和 | [回放](https://botlab.aiphys.cn/matches/e0b36ff6-5e56-44e2-b814-11c54445dc57) |
| 16 | v6 | 后手 | 3,5 | 0 | 胜 | [回放](https://botlab.aiphys.cn/matches/6e1d7a7b-cd97-465d-88b8-4d884c59b468) |
| 17 | v7 | 先手 | 5,3 | 0 | 和 | [回放](https://botlab.aiphys.cn/matches/82a1434b-3d04-4f74-8454-dcbfee039fd1) |
| 18 | v7 | 后手 | 1,7 | -1 | 负 | [回放](https://botlab.aiphys.cn/matches/81629b88-35cd-4f45-aa8b-da15a5c22908) |
| 19 | v6 | 先手 | 3,5 | 0 | 和 | [回放](https://botlab.aiphys.cn/matches/04ad8c4c-73a8-4687-b914-6bdd7cd14cad) |
| 20 | v6 | 后手 | 7,1 | -1 | 负 | [回放](https://botlab.aiphys.cn/matches/9bf22051-6d5a-4406-82d6-a6cad4486e3d) |

原始证据：`summary.json`、`frozen_versions.json`、`oracle_audit.json`；各对局子目录保存请求幂等键、创建响应、detail、live、日志和规则核验结果。所有统计来自这 20 个唯一比赛 ID。
