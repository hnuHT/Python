import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


# 残差块
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):  # use_1x1conv，是否使用1*1卷积层
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        # 使用1*1卷积层
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        # 假设用1*1卷积
        if self.conv3:
            X = self.conv3(X)  # 作用在X上面
        Y += X
        return F.relu(Y)


# 查看输入和输出形状一致的情况
blk = Residual(3, 3)
X = torch.rand(4, 3, 6, 6)
Y = blk(X)
print(Y.shape)
# torch.Size([4, 3, 6, 6])

# 也可以在增加输出通道数的同时，减半输出的高和宽。
blk = Residual(3, 6, use_1x1conv=True, strides=2)
print(blk(X).shape)
# torch.Size([4, 6, 3, 3])

# 第一层：ResNet的前两层跟之前介绍的GoogLeNet中的一样：
# 在输出通道数为64、步幅为2的7*7卷积层后，接步幅为2的3*3的最大汇聚层。
# 不同之处在于ResNet每个卷积层后增加了批量规范化层。
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


# 第一个模块
def resnet_block(input_channels, num_channels, num_residuals,  # 需要多少residuals块
                 first_block=False):  # 判断是否为第1个
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


# 接着在ResNet加入所有残差块，这里每个模块使用2个残差块
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))  # 是第1个，就不减半
# 下面就高宽减半，通道数加倍
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

# 与GoogLeNet一样，在ResNet中加入全局平均汇聚层，以及全连接层输出。
net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))


X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)


# 训练
lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
