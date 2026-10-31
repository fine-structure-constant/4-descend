本轮以 upload-v3 为对照，新增 upload-v4（导出目录 homework4/upload-v4-exact12）。旧版本保留。

新增 ≤12 枚棋子（双方合计前 12 手落子之后、第 13 手之前）的完整精确走法表：9,188,816 个非终局对称局面，逐局面记录精确分值（正数表示行棋方获胜且越大越快）与全部最优落子的 7 位列掩码。tools/movebook_build.cpp 离线链接 Pons 求解器与公开 7x6.book，先对全部 12 手局面 analyze，再自底向上做精确 negamax 得到 0～11 手的最优集合；tools/sort_movebook.py 再按对称键全局排序（不同手数的键会交错，必须全局有序才能二分查找）。表共 52.6 MiB，作为两个 uint8 缓冲区 movebook_keys / movebook_data 写入 model.pt，由 attach_opening.py 附加；support/movebook.py 只负责查表。

搜索在 ≤12 手直接取精确最优走法（有多个最优时用网络先验排序），不再消耗模拟次数；若精确分值为胜，则把"对方任意应手后仍必胜"的下界传播给 13 手以后的搜索。13 手以后保持 v3 的限时精确搜索与 MCTS。

验证：validate_movebook.py 对随机对局逐手检查覆盖、合法性、镜像一致性，并在 12 手局面调用 pons_solver analyze 复核分值与全部最优列；构建脚本另以 19,992 个随机局面复核 0～11 手推导值，0 处不一致。同网络对照（有表/无表，64 模拟，10 组交换先后手共 20 局）：14 胜 2 平 4 负（75%）。导出的整包按平台方式在临时目录加载 models/model.pt 实测两局，≤12 手全部与精确表一致，单手 CPU 最高 0.801 s。v3 棋谱回放：strong 局 56 个 ≤12 手决策中 10 个非最优，多数是"已败局面"或"较慢的胜局"（其中 12 手局面 1 个）；v4 现在一律取最快胜或保和的最优集合。

model.pt 约 59.5 MiB（含 6.9 MiB MIT 公开 12 手数据与 52.6 MiB 精确表），code.zip 约 288 KiB，均低于平台上限。最终参数 runs/cnn_replay/model_v4.pt；导出文件在 homework4/upload-v4-exact12/。

网络为从随机参数训练的残差 CNN：32 通道、4 个残差块，策略头与价值头；在原有我方、对方和空格通道之外，计算可落子、奇偶行及先后手特征。学习参数共 88,134 个。

训练先混合本地自对弈搜索样本、strong 棋谱、公开 UCI 开局价值和本地精确残局标签，再加强离线证明的中盘失误样本。817 个中盘样本由 teacher_exact.cpp 离线标注；本地编译程序只用于生成数据。上传推理使用 Python、NumPy 和 PyTorch。

model.pt 另包含 MIT 许可的 12 手公开开局数据缓冲区。这部分是搜索参考数据，不是神经网络学习参数。来源、许可及公开 UCI 数据说明见 REFERENCES.md 和 support/deep_opening.py。附加搜索参考由 attach_opening.py 写入参数文件。

推理保留 MCTS，并结合限时精确胜负搜索、置换表、空窗口探测和已证明胜负的传播。未知分支保留；短搜索的启发值不能直接用于剪枝。开局查询与整体搜索分别设有 CPU 预算。仍须以平台实际资源检查为准。

主要脚本：finetune_endgame.py、make_hard_data.py、train_hard_replay.py、teacher_exact.cpp；导出用 export.py；接口对局用 evaluate_api.py。登录信息仅由 BOTLAB_COOKIE 环境变量传入。

最终参数为 runs/cnn_replay/model.pt。评测棋谱与时间在 runs/final_v3_strong.json、runs/final_v3_standard.json；对照评测在 runs/previous_v2_strong_holdout.json。中间实验与这些最终结果分别保存。
