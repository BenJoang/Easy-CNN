# 03 训练与评估循环

对应 [`easy_cnn/engine.py`](../easy_cnn/engine.py)。负责用 `train_one_epoch()` 更新权重，并用 `evaluate()` 统计损失与准确率。

## 待补充

- 记录两条循环的输入、返回指标和模型模式。
- 用一批合成数据验证训练损失为有限值，且一次更新会改变模型参数。
