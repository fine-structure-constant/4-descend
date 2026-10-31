历史检查点的无损归档

本次清理将 70 份实验中间检查点改为 .pt.pack 文件。重复的大张量只在 .checkpoint_store/ 中保存一次；每份归档保留原文件的其他字节、SHA-256 和重建顺序。恢复后的 .pt 与原文件逐字节一致。归档和共享存储需要共同保留。

当前已交付的 upload* 文件夹不依赖该共享存储。实验目录中也保留了 v5/v6 最终模型和 v7 release_v7 最终模型的完整文件。源码、训练数据、对局记录和评估报告保留；Python 缓存、重复下载以及有排序成品的原始中间表已清理。

从 homework4 目录恢复某一旧检查点：

    python experiments/checkpoint_archive.py restore experiments/exact12_v7/runs/v7_train2/model.pt.pack

该命令恢复到原来的 model.pt 路径；已有不同文件时会拒绝覆盖。也可指定其他输出位置：

    python experiments/checkpoint_archive.py restore experiments/exact12_v7/runs/v7_train2/model.pt.pack --out /tmp/restored_model.pt

历史脚本若引用已归档的 .pt 路径，先恢复对应 .pt.pack 即可继续使用。全量清理清单及每份归档的原始摘要见 cleanup_report.json。

清理已对每份归档验证重建 SHA，并额外将 v7 最终参数来源 midbook_v5_nn2/model.pt 完整恢复到临时目录，验证其摘要与 upload-v7/model.pt 一致，再确认 PyTorch 能加载。
