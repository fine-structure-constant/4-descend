本目录以 upload-v4-exact12 为基线进行第 5 版训练，旧目录保持不变。

训练与推理分离：tools/pons_fast 与 label_positions.cpp 是保留许可证的 Pascal Pons 参考求解器，仅用于离线生成标签。平台 code.zip 只含 Python 和文本文件，不使用本地扩展、运行时编译或网络。公开评分数据作为不参与梯度更新的 model.pt 缓冲区，来源见 REFERENCES.md。

复现命令（在本目录执行）：

    g++ -O3 -std=c++17 -I tools/pons_fast tools/label_positions.cpp tools/pons_fast/Solver.cpp -o tools/label_positions
    python prepare_data.py --games 4500 --positions 30000 --workers 8 --seconds .16
    python train.py --out runs/v5 --steps 4000 --rounds 2 --games 24 --finetune-steps 800 --workers 6 --validate-every 400
    python validate_v5.py --model runs/v5/model.pt
    python export.py runs/v5/model.pt --out ../../upload-v5

train.py 从已有权重续训，校验并保留完整 exact12 动作表；混合精确中盘、精确残局、实战回放及少量开局样本。策略损失允许多个已证明最优动作，价值头学习局面胜负，辅助动作价值头学习已知的逐列结果；未求出的标签不当作和棋。训练阶段加入左右镜像，验证集合按镜像等价局面划分。

随后进行两轮混合自我对弈（包括历史版本），用离线求解器重新分析局面。训练和实战共用已证明的动作约束；噪声仅用于探索合法、未被精确排除的走法。best.pt 只按留出验证集选择，平台对战结果单独保存。

独立数据、CPU 检查和真实平台回放位于 runs/。训练 loss 或局面验证提升不等于稳定战胜 flash，最终结论以实际完成的对局为准。

实际结果：平台第 3 个版本已 READY，代码与权重摘要均与 upload-v5 一致。flash 8 局 3 胜 1 平 4 负（全部正常终局）；strong 8 局 4 胜 1 平 3 负。外部新种子且去除训练/镜像重合的 5,395 个局面，纯网络正确落子率由 80.39% 提升至 86.60%。自我对弈尝试了两轮，但最终按内部留出验证选中第 4,000 步精确监督权重，未用更差的后续权重替换。现阶段不能声称稳定战胜 flash 或 strong。完整可复查结果见 runs/release_manifest.json。
