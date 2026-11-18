from torch import Tensor, nn

class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.classifier = nn.Linear(8 * 14 * 14, 10)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.conv.weight, mode = "fan_in", nonlinearity="relu")
        nn.init.zeros_(self.conv.bias)

        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x:Tensor) -> Tensor:
        x = self.pool(self.relu(self.conv(x)))
        x = x.flatten(start_dim=1)
        return self.classifier(x)
    