import torch
from torch import nn
from d2l import torch as d2l
import time
import os

# -----------------------------
# Section 1. 命令式编程示例
# -----------------------------
def add(a, b):
    return a + b

def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g

print("命令式编程输出：", fancy_func(1, 2, 3, 4))  # 输出结果为 10

# -----------------------------
# Section 2. 模拟符号式编程：编译并执行字符串代码
# -----------------------------
def add_():
    return '''
def add(a, b):
    return a + b
'''

def fancy_func_():
    return '''
def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
'''

def evoke_():
    return add_() + fancy_func_() + 'print("符号式模拟输出：", fancy_func(1, 2, 3, 4))'

prog = evoke_()
print("\n【模拟符号式编程的程序源码】\n")
print(prog)

# 编译并执行该代码
y = compile(prog, '', 'exec')
exec(y)

# -----------------------------
# Section 3. 使用 PyTorch JIT 的混合式编程
# -----------------------------
# 定义一个简单的三层 MLP 网络
def get_net():
    net = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 2)
    )
    return net

# 随机输入一个 (1, 512) 的张量
x = torch.randn(size=(1, 512))

net = get_net()
print("\n【命令式运行网络输出】")
print(net(x))  # 命令式执行模型前向传播

# 使用 torch.jit.script 将其转换为可编译的模型
net_scripted = torch.jit.script(net)
print("\n【符号式运行网络输出】")
print(net_scripted(x))  # 符号式执行模型前向传播

# -----------------------------
# Section 4. 混合式编程性能对比：Benchmark 工具类
# -----------------------------
class Benchmark:
    """用于测量运行时间的工具类"""
    def __init__(self, description='Done'):
        self.description = description

    def __enter__(self):
        self.timer = d2l.Timer()
        return self

    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.4f} sec')

# 比较前后两种运行方式所需时间
print("\n【开始性能基准测试】")

net = get_net()
with Benchmark('命令式编程运行时间'):
    for _ in range(1000):
        net(x)

net_scripted = torch.jit.script(get_net())
with Benchmark('符号式编程运行时间（使用torchscript）'):
    for _ in range(1000):
        net_scripted(x)

# -----------------------------
# Section 5. 保存模型（序列化）
# -----------------------------
save_path = "my_mlp_scripted.pt"
net_scripted.save(save_path)  # 保存为 TorchScript 文件

print(f"\n模型已保存至: {save_path}")
if os.path.exists(save_path):
    file_size = os.path.getsize(save_path) / 1024
    print(f"保存文件大小: {file_size:.2f} KB")
else:
    print("保存失败，请检查路径或权限。")

