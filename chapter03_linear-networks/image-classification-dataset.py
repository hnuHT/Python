# 导入必要的库
import torch  # PyTorch 是主要的深度学习框架
import torchvision  # torchvision 提供了常见的数据集和图像变换工具
from torch.utils import data  # 用于加载小批量数据的工具（DataLoader 等）
from torchvision import transforms  # 图像预处理模块，如 ToTensor、Resize 等
from d2l import torch as d2l  # 引入 d2l 库的 PyTorch 接口，用于简化深度学习实验代码

# 使用 d2l 库中的函数设置图像显示为 SVG 格式（更清晰）
d2l.use_svg_display()

# 定义图像的预处理操作：将 PIL 图像或 ndarray 转换为张量（并自动归一化到 [0,1]）
trans = transforms.ToTensor()

# 下载并加载 Fashion-MNIST 训练集（train=True）并使用预处理操作 trans
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data",  # 数据下载和存放目录
    train=True,  # 指定为训练集
    transform=trans,  # 指定预处理函数（转 tensor）
    download=True  # 如果数据集不存在则自动下载
)

# 下载并加载 Fashion-MNIST 测试集（train=False）
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data",
    train=False,  # 指定为测试集
    transform=trans,
    download=True
)


# 将数字标签（0-9）转换成对应的文字标签
def get_fashion_mnist_labels(labels):
    # 标签列表，对应 Fashion-MNIST 的 10 个类别
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    # 遍历标签列表并转换为对应的字符串标签
    return [text_labels[int(i)] for i in labels]


# 显示多张图像的函数（支持标题、尺寸等设置）
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    # 设置图像显示的总大小
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)  # 创建子图
    axes = axes.flatten()  # 将子图对象展平成 1D 方便索引

    # 遍历图像与子图对象
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):  # 如果是张量，则先转为 NumPy 再显示
            ax.imshow(img.numpy())
        else:  # 否则直接显示
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)  # 不显示 x 轴坐标
        ax.axes.get_yaxis().set_visible(False)  # 不显示 y 轴坐标
        if titles:  # 如果提供了标题
            ax.set_title(titles[i])
    return axes


# 取一批训练集样本用于可视化
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))  # 随机取出一批大小为 18 的图像和标签
# 显示这批图像（将每张图像 reshape 成 28×28 大小），并显示对应的标签名
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
d2l.plt.show()  # 显示图像

# 设置每个 batch 的大小
batch_size = 256


# 获取用于 DataLoader 的工作线程数（一般设置为 CPU 核心数或更少）
def get_dataloader_workers():
    return 4  # 使用 4 个子线程读取数据（提高速度）


# 创建训练集的数据迭代器（DataLoader）
train_iter = data.DataLoader(
    mnist_train,  # 训练集数据
    batch_size,  # 每批次加载的数据数量
    shuffle=True,  # 打乱数据顺序（每个 epoch）
    num_workers=get_dataloader_workers()  # 使用多线程加速数据加载
)

# 测试从训练集中完整读取一轮（一个 epoch）所需时间
timer = d2l.Timer()  # 创建一个计时器
for X, y in train_iter:  # 遍历训练集的所有 batch
    continue  # 不做任何操作，只是计时
print(f'{timer.stop():2f} sec')  # 输出耗时（秒）


# 封装数据加载逻辑的函数，可自定义图像大小
def load_data_fashion_mnist(batch_size, resize=None):
    """
    下载 Fashion-MNIST 数据集并返回训练集和测试集的 DataLoader 迭代器。

    参数：
    - batch_size: 每个 batch 的大小
    - resize: 可选的图像尺寸，如 (64, 64)，如果不指定，则保持原始尺寸
    """
    # 创建变换列表：可包含 Resize 和 ToTensor
    trans = [transforms.ToTensor()]
    if resize:
        # 如果指定了 resize，则先进行 resize 再转 tensor
        trans.insert(0, transforms.Resize(resize))
    # 将变换操作组合成一个流水线
    trans = transforms.Compose(trans)

    # 加载并转换训练集
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    # 加载并转换测试集
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)

    # 返回训练集和测试集的 DataLoader
    return (
        data.DataLoader(mnist_train, batch_size, shuffle=True,
                        num_workers=get_dataloader_workers()),
        data.DataLoader(mnist_test, batch_size, shuffle=False,
                        num_workers=get_dataloader_workers())
    )
