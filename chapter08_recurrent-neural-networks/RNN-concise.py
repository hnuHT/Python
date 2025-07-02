import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# 1. 读取数据
batch_size, num_steps = 32, 35  # 每批大小32，序列长度为35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# 2. 定义超参数与RNN层
num_hiddens = 256  # 隐藏单元个数
rnn_layer = nn.RNN(input_size=len(vocab), hidden_size=num_hiddens)

# 3. 包装为完整的语言模型类
class RNNModel(nn.Module):  # @save
    """循环神经网络模型，封装了RNN层与输出层"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer  # RNN层
        self.vocab_size = vocab_size  # 词表大小
        self.num_hiddens = self.rnn.hidden_size  # 隐藏单元数

        # 判断是否为双向RNN
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, vocab_size)  # 输出层
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, vocab_size)

    def forward(self, inputs, state):
        # inputs形状: (batch_size, num_steps)
        # 转置并进行one-hot编码 → (num_steps, batch_size, vocab_size)
        X = F.one_hot(inputs.T.long(), self.vocab_size).type(torch.float32)
        Y, state = self.rnn(X, state)  # Y形状: (num_steps, batch_size, num_hiddens)
        # 将Y重塑为二维张量后接线性层输出
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        # 初始化隐状态（这里只支持单层单向RNN）
        return torch.zeros((self.num_directions * self.rnn.num_layers,
                            batch_size, self.num_hiddens), device=device)

# 4. 模型初始化与训练
device = d2l.try_gpu()
net = RNNModel(rnn_layer, vocab_size=len(vocab)).to(device)

# 5. 预测函数测试（随机初始化，效果差）
print(d2l.predict_ch8('time traveller', 10, net, vocab, device))

# 6. 模型训练
num_epochs, lr = 500, 1  # 训练500轮，学习率为1
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)

# 7. 模型预测输出
print(d2l.predict_ch8('time traveller', 50, net, vocab, device))
print(d2l.predict_ch8('traveller', 50, net, vocab, device))
