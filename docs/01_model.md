# 01 模型设计与测试

在 [`easy_cnn/model.py`](../easy_cnn/model.py) 中实现 `SmallCNN`。它接收一批 MNIST 灰度图，对每张图输出数字 `0–9` 对应的 10 个原始分数（logits）。

## 从张量形状推导网络

`B` 表示批大小。先推导每一层的输出，再用这些尺寸确定下一层的参数。

| 操作 | 输出形状 | 作用 |
| --- | --- | --- |
| 读取一批 MNIST | `[B, 1, 28, 28]` | 每张图片只有一个灰度通道 |
| `Conv2d(1, 8, kernel_size=3, padding=1)` | `[B, 8, 28, 28]` | 8 个卷积核提取局部特征，填充让宽高保持 `28×28` |
| `ReLU()` | `[B, 8, 28, 28]` | 将负值变为零，引入非线性 |
| `MaxPool2d(2)` | `[B, 8, 14, 14]` | 在每个 `2×2` 区域取最大值，宽高减半 |
| `flatten(start_dim=1)` | `[B, 1568]` | 保留批次维，把每张图的 `8 × 14 × 14` 个特征展平 |
| `Linear(1568, 10)` | `[B, 10]` | 为每张图输出 10 个分类分数 |

## 实现模型并初始化权重

1. 让 `SmallCNN` 继承 `nn.Module`。在 `__init__()` 开头调用 `super().__init__()`，再把各层赋给 `self.conv`、`self.relu`、`self.pool` 和 `self.classifier`。PyTorch 会登记这些子模块，优化器之后就能通过 `model.parameters()` 取得可训练参数。[PyTorch `nn.Module` 文档](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Module.html)
2. 在各层创建完成后调用一次 `self.reset_parameters()`。卷积后接 ReLU，卷积权重使用 `nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="relu")`。分类层权重使用 `nn.init.xavier_uniform_(self.classifier.weight)`。两层偏置都用 `nn.init.zeros_` 设为零。[PyTorch 初始化文档](https://docs.pytorch.org/docs/2.8/nn.init.html)
3. 在 `forward()` 中依次执行卷积、ReLU、池化、展平和分类。训练时通过 `model(images)` 得到 logits，再直接传给 `CrossEntropyLoss`。[PyTorch 交叉熵文档](https://docs.pytorch.org/docs/2.8/generated/torch.nn.CrossEntropyLoss.html)

如果创建 `SmallCNN()` 时遇到 `cannot assign module before Module.__init__() call`，先检查 `super().__init__()` 末尾的括号。括号会实际调用父类构造函数。

## 用 pytest 验证第一版

在 `tests/test_model.py` 中编写测试。用 CPU 上的合成张量验证模型接口。批大小 `1` 可以发现批次维意外丢失的问题，批大小 `4` 可以确认模型能同时处理多张图片。再用一个测试单独验证偏置初始化。[pytest 参数化指南](https://docs.pytest.org/en/stable/how-to/parametrize.html) · [PyTorch 张量断言](https://docs.pytorch.org/docs/2.8/testing.html)

```python
import pytest
import torch

from easy_cnn.model import SmallCNN


@pytest.mark.parametrize("batch_size", [1, 4])
def test_forward_returns_finite_logits(batch_size):
    model = SmallCNN().eval()
    images = torch.full((batch_size, 1, 28, 28), 0.5)

    with torch.inference_mode():
        logits = model(images)

    assert logits.shape == (batch_size, 10)
    assert torch.isfinite(logits).all().item()


def test_biases_start_at_zero():
    model = SmallCNN()
    for bias in (model.conv.bias, model.classifier.bias):
        torch.testing.assert_close(bias, torch.zeros_like(bias), rtol=0, atol=0)
```

在项目根目录用装有 PyTorch 和 pytest 的同一个 Python 环境运行。如果环境需要安装 pytest，可以运行 `python -m pip install pytest`。

```sh
python -m pytest -q tests/test_model.py
```

预期是 `3 passed`。通过后，可以把卷积输出通道数从 `8` 改成 `16`，先预测分类层输入尺寸，再用测试验证自己的推导。
