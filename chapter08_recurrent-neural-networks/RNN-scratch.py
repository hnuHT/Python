import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# ----------------------------
# 1. 加载数据
# ----------------------------
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# ----------------------------
# 2. 初始化模型参数
# ----------------------------
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    W_xh = normal((num_inputs, num_hiddens))  # 输入层到隐藏层权重
    W_hh = normal((num_hiddens, num_hiddens))  # 隐藏层循环权重
    b_h = torch.zeros(num_hiddens, device=device)  # 隐藏层偏置

    W_hq = normal((num_hiddens, num_outputs))  # 隐藏层到输出层权重
    b_q = torch.zeros(num_outputs, device=device)  # 输出层偏置

    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

# ----------------------------
# 3. 初始化RNN隐藏状态
# ----------------------------
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

# ----------------------------
# 4. RNN前向传播定义
# ----------------------------
def rnn(inputs, state, params):
    # inputs: (时间步数, 批量大小, 词表大小)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state  # 当前隐藏状态
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)

# ----------------------------
# 5. 模型类封装
# ----------------------------
class RNNModelScratch:
    """从零实现的RNN字符级语言模型"""
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state = init_state
        self.forward_fn = forward_fn

    def __call__(self, X, state):
        # 独热编码后转为float32并转置：时间步为最外层
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)

# ----------------------------
# 6. 预测函数：给定前缀生成字符
# ----------------------------
def predict_ch8(prefix, num_preds, net, vocab, device):
    """生成prefix后num_preds个字符"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]  # 用索引保存结果
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))

    with torch.no_grad():
        for y in prefix[1:]:  # 预热
            _, state = net(get_input(), state)
            outputs.append(vocab[y])
        for _ in range(num_preds):  # 生成字符
            y, state = net(get_input(), state)
            outputs.append(int(y.argmax(dim=1).reshape(1)))  # 注意reshape(1)

    return ''.join([vocab.idx_to_token[i] for i in outputs])

# ----------------------------
# 7. 梯度裁剪
# ----------------------------
def grad_clipping(net, theta):
    """裁剪梯度，防止梯度爆炸"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

# ----------------------------
# 8. 单个训练周期
# ----------------------------
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 累加损失总和与词元数
    for X, Y in train_iter:
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(state, tuple):
                state = tuple(s.detach() for s in state)  # 更安全的写法
            else:
                state = state.detach()

        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()

        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

# ----------------------------
# 9. 训练主函数
# ----------------------------
def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity', legend=['train'], xlim=[10, num_epochs])

    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)

    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)

    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))  # 打印预测结果
        animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, 速度 {speed:.1f} 词元/秒 on {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))

# ----------------------------
# 10. 初始化并训练模型
# ----------------------------
num_hiddens = 512
num_epochs, lr = 500, 1
device = d2l.try_gpu()

# 第一次：顺序采样训练
net = RNNModelScratch(len(vocab), num_hiddens, device, get_params, init_rnn_state, rnn)
train_ch8(net, train_iter, vocab, lr, num_epochs, device)

# 再加载一次数据，防止 train_iter 用尽
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# 第二次：使用随机采样训练
net = RNNModelScratch(len(vocab), num_hiddens, device, get_params, init_rnn_state, rnn)
train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=True)
