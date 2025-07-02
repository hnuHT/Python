import torch
from torch import nn
from d2l import torch as d2l

# 1. 读取数据
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# 2. 定义模型
vocab_size = len(vocab)
num_hiddens = 256
num_inputs = vocab_size
device = d2l.try_gpu()

gru_layer = nn.GRU(input_size=num_inputs, hidden_size=num_hiddens)
model = d2l.RNNModel(gru_layer, vocab_size)
model = model.to(device)

# 3. 训练模型
num_epochs, lr = 500, 1
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
