# 版本记录

本文件是 v1–v8 与 v10 的统一索引。每个版本的代码包保存在 `upload-vN/code.zip`，权重 `model.pt` 保留在本地但不纳入 git；下表的 SHA-256 用于校验。详细说明见 `docs/releases/vN.md`，机器可读清单见 `docs/releases/vN.json`。

平台版本号、UUID 与 READY 状态来自历史快照（见 `agent.md`），上传或指定参赛前应重新查询。本地版本号与平台版本 number 不一致，比较时以目录和 UUID 为准。

## 版本一览

| 标签 | 发布目录 | 类型 | 平台状态 | 详细说明 |
|---|---|---|---|---|
| `v1` | `upload-v1` | 基础 AlphaZero 流程：小型双层 MLP + 基础 MCTS | 未确认 | [`v1.md`](docs/releases/v1.md) |
| `v2` | `upload-v2` | 256 宽残差 MLP，专家/strong 数据与战术搜索 | 未确认 | [`v2.md`](docs/releases/v2.md) |
| `v3` | `upload-v3` | 32 通道、4 残差块 CNN，重力/奇偶特征，残局搜索及开局参考数据 | number 1 · READY | [`v3.md`](docs/releases/v3.md) |
| `v4` | `upload-v4-exact12` | v3 CNN + 完整 ≤12 子精确动作表，纯 Python 推理 | number 2 · READY | [`v4.md`](docs/releases/v4.md) |
| `v5` | `upload-v5` | 精确监督 CNN + 逐列动作价值辅助头，精确数据重分析，Pons 深开局参考 | number 3 · READY | [`v5.md`](docs/releases/v5.md) |
| `v6` | `upload-v6` | 32 通道，窄残局及顶端备用步完整求解、风险损失、972 条反事实纠错 | number 4 · READY | [`v6.md`](docs/releases/v6.md) |
| `v7` | `upload-v7` | 48 通道、195,517 参数，41,661 个逐列完整 WDL 中盘记谱局面 | number 5 · READY | [`v7.md`](docs/releases/v7.md) |
| `v8` | `upload-v8` | v7 原权重 + Rust 离线精确求解常驻进程，失败回退 Python | 未确认 | [`v8.md`](docs/releases/v8.md) |
| `v10` | `upload-v10` | 纯 CNN（6 平面、192 通道、12 残差块）+ PUCT，18 轮自对弈，7,989,402 参数 | 未确认 | [`v10.md`](docs/releases/v10.md) |

## 产物摘要

| 版本 | code.zip 字节 | code.zip SHA-256 | model.pt 字节 | model.pt SHA-256 |
|---|---:|---|---:|---|
| v1 | 6,242 | `2300115403a58a840ed394c557166940334b619b2abbb6c21ba4883a56219d5e` | 137,768 | `97aedc7f200043a763e0da27990629b607a22a37b5a60257edcb335a1afee050` |
| v2 | 9,670 | `29247f21b89380463c8ac5d6e838bf801878c27f3119ea5ab7b355bb751238cc` | 733,730 | `4a2f7e8e85ffe0e27aa0b3bab547dddda7b35bd0cb8de108e7efc68fa0047a0d` |
| v3 | 291,934 | `92b0cdb0eb69c762b04c4994e16de01654a2e5c231eb878c6a4627b3513c5c5e` | 7,303,672 | `1ecec1bd751dc3bdbefdf3f800b618ec0dc51ce1c766c6d9be8b5a3e15f0c03b` |
| v4 | 294,229 | `20822b942bad07aadfa6d1a65784e5e14a65399d8435d9783ed4bd4adc994769` | 62,438,682 | `9d5f91124bb8849aa71f85e935cec73e903966e2439fe41f3552db3aadfc6fbf` |
| v5 | 309,807 | `69a081af66f964939ea157d14f353f31211483df1793c327e40ea49e0f8daed6` | 96,029,983 | `913bb04165cfad3e5e53fd649454f2cfdfb3d2cdad93e6f401d89a543550ae36` |
| v6 | 322,414 | `6d47ad6406690071a39f2991c395b6cace7673cb1a0e93e88e52cd01ce6408bc` | 96,029,983 | `69641c9e5b57950bf5930f491a040d6ae490f879f950c327c91db884944634bb` |
| v7 | 324,286 | `9539999708ec298d32c304b270b17973ad4411843e3b5359405b51628b086656` | 97,503,563 | `2a98dbb2d08d6c6c4b341e3d0746f3c8193541622e422c4ce05293a0228472ec` |
| v8 | 852,203 | `d5ff1abcfb19859d707fcb5ab59de1ca9605591094ad3521806cac1284482380` | 97,503,563 | `2a98dbb2d08d6c6c4b341e3d0746f3c8193541622e422c4ce05293a0228472ec` |
| v10 | 5,008 | `9edf958d54f1bcc16416051444959174385acdabf04c9452a6aea891d4962c5b` | 31,954,402 | `d96b1f0afe1b41e4c6f73ad20b6452e8065776d5a10ed9ac630fcd1f9c0cc0c6` |

`code.zip` 纳入 git，`model.pt` 不纳入；本表摘要用于确认本地权重与历史交付一致。

## Git 标签

`v1` 到 `v8` 与 `v10` 为附注标签，分别指向加入对应版本代码包的提交。查看某个版本的提交与代码包：

```bash
git show v7 --stat
git tag -l 'v*'
```

## 证据边界

- 历史对局、开发集对照和本地求解器验证不能直接等同于平台实战强度，具体限制写在各版本说明的“注意事项”。
- 原生 v4 实验与纯 Python v4-exact12 是两个不同交付，不能混用强度结论。
- v8 只确认本地打包与验证，没有上传成功记录。
- v10 是纯 CNN 路线，没有精确表或求解器兜底；官方 play 接口的 40 局结果不是平台排名，也没有上传记录。
