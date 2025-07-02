import torch
from torch import nn

# ------------------------------
# 定义一个通用的卷积测试函数，用于输出结果形状
def comp_conv2d(conv2d, X):
    # 把二维图像 X 扩展为四维：[batch_size=1, channel=1, height, width]
    X = X.reshape((1, 1) + X.shape)
    # 使用给定的 conv2d 层进行前向传播
    Y = conv2d(X)
    # 仅返回输出的高度和宽度（去掉批量和通道）
    return Y.reshape(Y.shape[2:])

# ------------------------------
# 示例 1：对称填充
# 卷积核大小为 3×3，padding=1，相当于在四周各填一圈，保持尺寸不变
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))  # 输入大小为 8×8
print(comp_conv2d(conv2d, X).shape)  # 输出 torch.Size([8, 8])

# ------------------------------
# 示例 2：不对称填充
# 卷积核大小为 5×3，上下 padding=2，左右 padding=1
# 保证输出尺寸仍为 8×8：
# H_out = (8 + 2*2 - 5)/1 + 1 = 8
# W_out = (8 + 2*1 - 3)/1 + 1 = 8
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
print(comp_conv2d(conv2d, X).shape)  # 输出 torch.Size([8, 8])

# ------------------------------
# 示例 3：使用步幅（stride）
# 核为 3×3，padding=1，stride=2
# 每次滑动跨 2 个像素，所以输出尺寸减半：
# H_out = (8 + 2*1 - 3)/2 + 1 = 4
# W_out = 同上 = 4
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
print(comp_conv2d(conv2d, X).shape)  # 输出 torch.Size([4, 4])

# ------------------------------
# 示例 4：不对称 kernel、padding、stride 同时变化
# 卷积核大小为 3×5，高 padding=0，宽 padding=1
# stride=(3,4)：高方向每 3 步滑动一次，宽方向每 4 步滑动一次
# 输出尺寸：
# H_out = (8 + 2*0 - 3)/3 + 1 = 2
# W_out = (8 + 2*1 - 5)/4 + 1 = 2
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
print(comp_conv2d(conv2d, X).shape)  # 输出 torch.Size([2, 2])
