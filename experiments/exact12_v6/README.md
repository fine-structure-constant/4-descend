v6：备用步残局求解与精确反事实纠错训练

基线：upload-v5。旧版上传包及训练目录保持不变。最终输出为 homework4/upload-v6，只含 code.zip 和 model.pt。

推理：前十二手精确动作表；对三个主要空列（每列至少两格）加任意顶端单格的残局优先完整求解。所有实际落点均保留。只有一格被证明不可能参与任一方四连时，才将其与其他同类格合并为节奏步，并保留剩余步数与先后手。CPU 超时返回未知边界，不把启发式估值当证明。求解得到胜着或安全和棋动作后，网络仅在相同证明等级的动作之间选取。

中盘：降低无支撑三连的启发式权重，使用更快的局面评估完成更深的搜索。另有 972 条来自此前实战及反事实应手、经求解器证明的可胜/可和纠错条目，避免已知失误在权重更新后回归。该表由 build_corrections.py 生成，属于选择性记谱，不计为网络的泛化能力。外部测试集逐一确认没有命中这些条目，新 v6 平台对局未用于训练或记谱。

训练：44,843 个去重局面包含新增 15,000 个窄残局与过渡局面；再加入 923 个实战反事实局面，最终 45,716 个去重局面。先比较 32/48 通道，随后用选定小网络纠错续训。两种结构各尝试 12 局混合自博弈，终局后对所有落子前局面进行精确重分析。最优检查点按内部留出集选择，不默认选最后一轮。

风险损失为预测动作概率加权的 max(0, V*(s)-Q*(s,a))。只有已证明的标签参与该损失。可胜走成和/负、可和走成负都会受到惩罚；原本所有走法均负时不追加动作惩罚。价值头学习精确胜负，辅助动作价值头学习逐列结果。训练梯度检查覆盖上述区别。

网络对比：32 通道 97,549 个参数；48 通道 195,517 个参数。48 通道验证正确率只多约 0.20 个百分点，单次推理慢约 23%，综合收益不足本轮阈值，因此保留 32 通道。两份训练检查点均保存于 runs/small 和 runs/wide。

复现（本目录）：
    g++ -O3 -std=c++17 -I tools/pons_fast tools/label_positions.cpp tools/pons_fast/Solver.cpp -o tools/label_positions
    python prepare_constrained.py
    python train.py --channels 32 --out runs/new_small --steps 1800 --rounds 1 --games 12 --finetune-steps 400 --validate-every 300 --lr .00012 --seed 2026103117
    python prepare_refutations.py
    python train.py --model runs/new_small/model.pt --channels 32 --out runs/new_refined --data data/refinement.npz --steps 900 --rounds 0 --validate-every 150 --lr .00008 --seed 2026103122
    python build_corrections.py
    python export.py runs/new_refined/model.pt --out ../../upload-v6-new
    python validate_package.py ../../upload-v6-new

仅生成离线标签的参考 C++ 求解器及公开评分数据来源见 REFERENCES.md 和 tools/AGPL-3.0.txt。平台包只含 Python/文本文件，无运行时编译、本地扩展或网络调用。训练数据、检查点和详细日志保留在本目录。

验证材料：
    runs/constrained_validation.json：10,000 个窄残局，与另一个求解器的证明标签交叉校验；全部求得最优结果，另有 300 次关闭节奏步合并的等价检查。
    runs/tempo_impact.json：458 个只允许顶端单格最优的局面，其中 312 个没有眼前一步胜或立即挡杀。
    runs/v5_actual_diagnosis.json：v5 实战窄残局的已知可避免降级为 0；已证失误更早发生在第 17 手。
    runs/v5_tempo_ablation.json、v6_tempo_ablation.json：同一组 120 个关键节奏局面均正确，平均 CPU 由 19.55ms 降至 1.80ms。
    runs/architecture_comparison.json：网络扩容实测。
    runs/training_validation.json：风险损失与非法动作梯度检查。
    runs/final_holdout.json：外部 9,570 个局面的纯网络比较，完全去除训练及镜像重合。
    runs/package_validation.json：真实上传包独立解压、加载及备用步检查。
    runs/flash_probe：平台 flash 对战原始记录。
    runs/strong_protocol.json、strong_openings.json：新种子生成 20 组八手理论和棋开局，交换先后手共 40 局；当前对局结果不反向更新候选。

最终实测：平台第 4 个版本检查通过，SHA 与 upload-v6 一致。strong 新 20 组八手理论和棋开局换色 40 局：23 胜 12 平 5 负，得分 72.5%，配对 bootstrap 95% 区间 [61.25%,83.75%]；20 组为 12 胜 6 平 2 负，未达到预设 70% 配对获胜门槛，因此不宣称全部稳定性判据通过。flash 共 8 局：平台结果 3 胜 3 平 2 负，其中正常终局 2 胜 3 平 2 负，另 1 胜因 flash 超时；该局超时前双方的精确结果为和棋。所有落子帧和终局已独立核验，无我方超时。
外部新 2,789 个窄残局在清空置换表的条件下均选择最优动作，平均 4.57ms，最高 0.642 秒 CPU。纯网络外部 9,570 局面正确率 86.92%→88.88%；过渡到窄残局前的四主列子集 82.54%→88.31%。正式 strong 测试中精确纠错表命中 0 次，所有测试期间候选冻结。
