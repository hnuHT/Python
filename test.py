import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 不断开图
y1 = x * x        # y1 = x^2
z1 = y1 * x       # z1 = x^3
z1.sum().backward()
print("不使用 detach()，x.grad:", x.grad.clone())

x.grad.zero_()

# 使用 detach()
y2 = x * x
u2 = y2.detach()  # detach 断开图
z2 = u2 * x       # z2 = u2 * x，当 u2 是常数
z2.sum().backward()
print("使用 detach()，x.grad:", x.grad)
