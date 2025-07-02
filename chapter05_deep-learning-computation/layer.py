import torch
from torch import nn
from torch.nn import functional as F

# ============================
# 定义输入：一个 2x20 的随机张量
# 其中 2 是 batch size，20 是输入维度
# ============================
X = torch.rand(2, 20)

# ============================
# 定义一个多层感知机（MLP），它是nn.Module的子类
# 包含两个全连接层：Linear(20 -> 256) + ReLU + Linear(256 -> 10)
# ============================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()  # 初始化父类
        self.hidden = nn.Linear(20, 256)  # 第一个线性层（隐藏层）
        self.out = nn.Linear(256, 10)     # 第二个线性层（输出层）

    def forward(self, X):
        # 前向传播：隐藏层 -> ReLU 激活 -> 输出层
        return self.out(F.relu(self.hidden(X)))


# ============================
# 自定义实现 nn.Sequential 的类
# 作用：像 nn.Sequential 一样按顺序叠加多个模块
# ============================
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        # 遍历所有传入的子模块，并按添加顺序注册到 _modules 字典中
        for idx, module in enumerate(args):
            self._modules[str(idx)] = module  # OrderedDict 自动管理顺序

    def forward(self, X):
        # 依次将输入 X 通过每个子模块
        for block in self._modules.values():
            X = block(X)
        return X


# ============================
# 更复杂的模块：展示固定参数、控制流、参数共享等特性
# ============================
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 创建一个不可训练的随机矩阵 rand_weight（不参与反向传播）
        self.rand_weight = torch.rand((20, 20), requires_grad=False)

        # 定义一个共享的全连接层：Linear(20 -> 20)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        # 第一次通过线性层
        X = self.linear(X)

        # 用固定矩阵进行线性变换（矩阵乘法） + 偏移 + ReLU 激活
        X = F.relu(torch.mm(X, self.rand_weight) + 1)

        # 再次使用相同的线性层（参数共享）
        X = self.linear(X)

        # 控制流：如果输出值太大，就缩小它，直到总绝对值不超过 1
        while X.abs().sum() > 1:
            X /= 2

        # 返回所有输出元素的总和（结果是一个标量）
        return X.sum()


# ============================
# 实例化 FixedHiddenMLP 并运行
# ============================
net = FixedHiddenMLP()
print(net(X))  # 输出一个标量
