# 02 数据加载

对应 [`easy_cnn/data.py`](../easy_cnn/data.py)。负责从 `data/` 读取 MNIST、转换图片、划分训练集和验证集，以及创建 `DataLoader`。

## 待补充

- 记录 `build_loaders()` 的参数、返回值和数据划分的随机种子。
- 验证一批图片的形状为 `[B, 1, 28, 28]`，标签的形状为 `[B]` 且类型为 `torch.long`。
