import torch
from torch import nn
from d2l import torch as d2l

# ------------------------
# 自定义 Dropout 函数
# ------------------------
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    # 随机掩码，大于dropout的置为1，其他为0
    mask = (torch.randn(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)  # 保持期望不变


# ------------------------
# 手动实现 MLP + Dropout
# ------------------------
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
dropout1, dropout2 = 0.0, 0.5  # 第一层不drop，第二层drop一半

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        X = X.reshape((-1, self.num_inputs))         # flatten
        H1 = self.relu(self.lin1(X))                 # 第1层+ReLU
        if self.training:
            H1 = dropout_layer(H1, dropout1)         # dropout1
        H2 = self.relu(self.lin2(H1))                # 第2层+ReLU
        if self.training:
            H2 = dropout_layer(H2, dropout2)         # dropout2
        return self.lin3(H2)                         # 输出层

# 实例化模型、训练参数
net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)

# 训练模型
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.show()


# ------------------------
# 简洁实现 MLP + Dropout
# ------------------------
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256), nn.ReLU(),
    nn.Dropout(dropout1),               # dropout 层1
    nn.Linear(256, 256), nn.ReLU(),
    nn.Dropout(dropout2),               # dropout 层2
    nn.Linear(256, 10)
)

# 初始化权重
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

# 训练
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.show()
