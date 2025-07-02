import os
import time
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------
# Section 1. 设置设备（使用GPU）
# -----------------------------
device = d2l.try_gpu()  # 自动选择GPU，如果没有则使用CPU
print(f"\n当前使用的设备: {device}")

# -----------------------------
# Section 2. NumPy vs PyTorch 基础矩阵乘法性能对比
# -----------------------------

# NumPy在CPU上进行随机矩阵乘法（阻塞）
with d2l.Benchmark('NumPy矩阵乘法 (CPU)'):
    for _ in range(10):
        a_np = np.random.normal(size=(1000, 1000))
        b_np = np.dot(a_np, a_np)

# PyTorch在GPU上进行相同大小的张量乘法（异步）
with d2l.Benchmark('PyTorch矩阵乘法 (GPU 异步)'):
    for _ in range(10):
        a_torch = torch.randn(size=(1000, 1000), device=device)
        b_torch = torch.mm(a_torch, a_torch)

# 强制PyTorch完成所有排队任务（同步）
with d2l.Benchmark('PyTorch矩阵乘法 (GPU 同步)'):
    for _ in range(10):
        a_torch = torch.randn(size=(1000, 1000), device=device)
        b_torch = torch.mm(a_torch, a_torch)
    torch.cuda.synchronize(device)

# -----------------------------
# Section 3. 异步执行示意：操作被排队
# -----------------------------
x = torch.ones((1, 2), device=device)
y = torch.ones((1, 2), device=device)
z = x * y + 2
print("\n异步执行示例输出 z =", z)

# -----------------------------
# Section 4. 绘图模拟：前端-后端交互机制
# -----------------------------

def draw_async_flow():
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # 绘制前端线程的任务发出
    ax.arrow(0.5, 3, 1.5, 0, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
    ax.text(0.3, 3, '前端发出任务', color='blue')

    # 后端执行任务
    ax.add_patch(plt.Rectangle((2.1, 2.7), 2.5, 0.5, color='lightgray'))
    ax.text(2.3, 2.85, '后端执行 x * y', fontsize=10)

    ax.add_patch(plt.Rectangle((4.7, 2.7), 2.5, 0.5, color='lightgray'))
    ax.text(5.0, 2.85, '后端执行 + 2', fontsize=10)

    # 最终结果返回
    ax.arrow(7.3, 3, 1.5, 0, head_width=0.2, head_length=0.2, fc='green', ec='green')
    ax.text(8.0, 3.2, '返回结果 z', color='green')

    # 线程之间的分离
    ax.text(2.5, 1.5, '前端 Python 线程', fontsize=10, color='blue')
    ax.text(2.5, 0.7, '后端 C++ 线程池', fontsize=10, color='gray')
    ax.plot([0.5, 9], [2.4, 2.4], linestyle='--', color='black', linewidth=0.5)

    plt.title("图示：PyTorch 异步计算任务在前端与后端的交互", fontsize=12)
    plt.tight_layout()
    plt.show()

draw_async_flow()
