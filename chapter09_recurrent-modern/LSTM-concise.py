import torch
from torch import nn
from d2l import torch as d2l

# ===============================
# 1. 加载时间机器数据集
# ===============================
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# ===============================
# 2. 定义模型
# ===============================
vocab_size = len(vocab)      # 词表大小作为输入和输出的维度
num_hiddens = 256            # 隐藏单元数量
num_inputs = vocab_size      # 每个时间步的输入为one-hot向量，维度等于词表大小
device = d2l.try_gpu()       # 自动选择GPU或CPU

# 创建一个LSTM层：输入为one-hot编码，输出为num_hiddens维隐状态
lstm_layer = nn.LSTM(input_size=num_inputs, hidden_size=num_hiddens)

# 封装成 RNNModel，内部会添加一个输出层将hidden_state映射到词表大小
model = d2l.RNNModel(lstm_layer, vocab_size)
model = model.to(device)

# ===============================
# 3. 训练模型
# ===============================
num_epochs, lr = 500, 1
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
