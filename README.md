# Easy CNN

用 PyTorch 从随机初始化开始训练一个小型卷积神经网络，识别 MNIST 中的单个手写数字（0–9）。目前已完成数据准备，模型和训练代码待实现。

## 项目结构

```text
.
├── README.md
├── .gitignore
└── data/
    └── MNIST/
        └── raw/  # 下载后生成，包含训练集和测试集
```

`data/` 由 Git 忽略，克隆仓库后可按下面的步骤下载数据。

## 准备环境和数据

1. 在 Python 环境中安装 `torch` 和 `torchvision`。按操作系统和计算平台（CPU 或 CUDA）选择安装命令。[PyTorch 官方安装页](https://pytorch.org/get-started/locally/)
2. 在项目根目录运行下方命令。Torchvision 会下载并解压 MNIST 的 60,000 张训练图和 10,000 张测试图及其标签。已有数据会直接复用。[Torchvision MNIST 文档](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html)

```sh
python -c "from torchvision.datasets import MNIST; MNIST(root='data', train=True, download=True); MNIST(root='data', train=False, download=True)"
```

后续编写训练代码时，使用 `root='data'` 读取数据。`train=True` 读取训练集，`train=False` 读取测试集。
