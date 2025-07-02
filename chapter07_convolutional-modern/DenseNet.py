import torch
from torch import nn
from d2l import torch as d2l

# 卷积块（BN -> ReLU -> 3x3Conv）
def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1)
    )

# 稠密块（多个卷积块 + 拼接）
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super().__init__()
        layers = []
        for i in range(num_convs):
            layers.append(conv_block(input_channels + i * num_channels, num_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)  # 在通道维度拼接
        return X

# 过渡层（压缩通道 + 降低高宽）
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )

# 第一层：卷积+BN+ReLU+池化（和ResNet一致）
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

# 稠密块和过渡层的堆叠
num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]
blks = []

for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    num_channels += num_convs * growth_rate  # 更新通道数
    if i != len(num_convs_in_dense_blocks) - 1:  # 最后一个块后不加过渡层
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2

# 拼接所有部分，构造完整网络
net = nn.Sequential(
    b1, *blks,
    nn.BatchNorm2d(num_channels), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
    nn.Linear(num_channels, 10)
)

# 训练
lr, num_epochs, batch_size = 0.1, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
