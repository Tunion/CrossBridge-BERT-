# CrossBridge-BERT-
CrossBridge-BERT 是一个面向跨链场景的文本二分类模型：把跨链交易序列模式或原生数据模式整理成文本后，送入 BERT 做监督微调，输出“是否异常”的预测及异常概率，用于告警与人工复核。训练脚本用 HuggingFace Trainer 进行评估与早停，保存最优模型；推理脚本加载已训模型，对 CSV 批量推断并写出 pred 与 prob_abnormal  
