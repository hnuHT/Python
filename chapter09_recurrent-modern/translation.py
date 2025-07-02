import os
import torch
from d2l import torch as d2l

# 下载和提取“英语-法语”数据集（Tatoeba 项目）
# 该数据集由英语和法语的句子对组成，每一行为一对句子，用制表符分隔
d2l.DATA_HUB['fra-eng'] = (
    d2l.DATA_URL + 'fra-eng.zip',
    '94646ad1522d915e7b0f9296181140edcf86a4f5'
)

# 读取原始数据（返回整个文本内容）
def read_data_nmt():
    """载入'英语－法语'数据集"""
    data_dir = d2l.download_extract('fra-eng')  # 解压到默认目录
    with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f:
        return f.read()


# 文本预处理：去除特殊空格、转小写、处理标点分隔
def preprocess_nmt(text):
    """预处理'英语－法语'数据集"""

    def no_space(char, prev_char):
        # 判断当前字符是标点且前面没有空格，需要加空格
        return char in set(',.!?') and prev_char != ' '

    # 替换不间断空格（\u202f、\xa0），并全部转为小写
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()

    # 在标点前添加空格，以便分词时将其单独分开
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)


# 词元化函数：将句子对按空格切分为单词列表
def tokenize_nmt(text, num_examples=None):
    """词元化'英语－法语'数据集"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i >= num_examples:
            break
        parts = line.split('\t')  # 制表符分隔的句子对
        if len(parts) == 2:
            source.append(parts[0].split(' '))  # 英语句子
            target.append(parts[1].split(' '))  # 法语句子
    return source, target


# 绘制源语言和目标语言句子的词元数分布直方图
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """绘制词元数量的直方图（用于源/目标序列）"""
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')  # 设置目标语言的图案为斜线
    d2l.plt.legend(legend)
    d2l.plt.show()


# 截断或填充句子到指定长度
def truncate_pad(line, num_steps, padding_token):
    """将序列截断或填充到固定长度"""
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))


# 将词元序列转换为张量（添加 <eos> 结尾，处理填充）
def build_array_nmt(lines, vocab, num_steps):
    """将文本序列转换为张量形式并记录有效长度"""
    # 将词元序列转换为索引序列
    lines = [vocab[l] for l in lines]
    # 在句子末尾添加结束符
    lines = [l + [vocab['<eos>']] for l in lines]
    # 每个句子截断或填充为 num_steps 长度
    array = torch.tensor([
        truncate_pad(l, num_steps, vocab['<pad>']) for l in lines
    ])
    # 有效长度统计（不含 pad）
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len


# 最终加载数据集（组合前面步骤）
def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回数据迭代器和词表（src_vocab, tgt_vocab）"""
    # 加载和预处理文本
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)

    # 构建词表，移除出现频率 < 2 的词元
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])

    # 构建张量和有效长度
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)

    # 创建小批量迭代器
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)

    return data_iter, src_vocab, tgt_vocab


# ========================
# 测试功能（仅执行一次）
# ========================
if __name__ == '__main__':
    # 加载数据（小批量大小为2，序列长度为8）
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)

    # 打印第一个小批量内容
    for X, X_valid_len, Y, Y_valid_len in train_iter:
        print('X (source):', X.type(torch.int32))
        print('X的有效长度:', X_valid_len)
        print('Y (target):', Y.type(torch.int32))
        print('Y的有效长度:', Y_valid_len)
        break
