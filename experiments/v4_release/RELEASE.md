# v4 release

包大小使用更严格的十进制上限检查：code.zip < 5,000,000 字节，model.pt < 100,000,000 字节。

“前12手”指双方合计前12次落子。对落子前有0～11枚棋子的全部3356580个对称去重局面，记谱表完整存储胜／平／负意义下的最优动作集合。最短取胜步数不作为次级目标。若有多个最优动作，CNN可以在最优集合内选择。第13手起会调用CNN，再结合限时精确搜索和MCTS决策。

模型文件包括CNN学习参数与棋谱数据缓冲区。公开资料、许可证和实现来源见REFERENCES.md。推理无需联网或编译；包内加速模块适用于Linux x86_64、CPython 3.8+（abi3），GLIBC最低符号版本为2.17。

核验文件：

- runs/all_opening_verification.txt：对全部前12手局面的覆盖、合法性和最优性逐项核验。
- runs/all_opening_independent_dtm_verification.txt：使用另一份完整精确结果再次核验。
- runs/native_search_validation.txt：独立数组棋盘规则及残局检查。
- runs/native_boundary_validation.json：开局／模型切换边界及资源用量。
- runs/stability_openings.json：测试前固定的20组开局；8组常见开局及12组按固定种子抽取的理论和局开局。
- runs/strong_stability_batch1.json、batch2.json：先后手各半，共40局官方strong实测；未完成对局不计入胜负。
- runs/stability_summary.json：按开局配对统计及bootstrap区间。

稳定性判据预先固定为：至少40局，总得分率≥70%，成对比赛获胜率≥70%，按开局配对重采样的95%区间下界>50%，每手CPU时间<1秒。正式结果以实际对局报告为准。

复现主要步骤：编译tools/build_wdl_movebook.cpp（-Itools/pons_fast，链接Solver.cpp），整理排序，再运行attach_opening.py把棋谱附加到本地训练的CNN参数，最后用export.py导出。原始公开数据可从REFERENCES.md链接获取。所有最终对局报告保存推理代码与权重的SHA-256。

本次完成的40局结果：30胜9平1负，得分率86.25%；先手18胜2平、后手12胜7平1负。20组比赛18胜2平0负。按开局配对bootstrap的95%区间为约[78.75%,92.53%]。实测单手CPU最高0.5211秒。判据通过。官方接口中途错误经重试后完成，错误响应未算成胜局。

平台实测补充：upload-v4 的 code.zip 上传被 HTTP 400 / INVALID_PACKAGE 拒绝（代码包需要 Python 或文本文件）。包内本地编译扩展及 C++ 文件不符合上传格式；先前独立加载与大小核验不代表平台接收。upload-v4-exact12 是另一份已接收的纯 Python 版本，其 flash 实测见 runs/flash_probe。
