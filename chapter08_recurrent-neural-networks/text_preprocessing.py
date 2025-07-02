import collections
import re
from d2l import torch as d2l

# 注册数据集下载链接
d2l.DATA_HUB['time_machine'] = (
    d2l.DATA_URL + 'timemachine.txt',
    '090b5e7e70c295757f55df93cb0a180b9691891a'
)

# 读取文本文件，清洗标点和大小写
def read_time_machine():  # @save
    """将时间机器数据集加载为文本行的列表"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

# 词元化：按单词或字符拆分
def tokenize(lines, token='word'):  # @save
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        raise ValueError('未知词元类型：' + token)

# 统计词频
def count_corpus(tokens):  # @save
    """统计词元的频率"""
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

# 构建词表
class Vocab:  # @save
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []

        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        # 构建索引映射表
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

# 综合封装：从原始数据到索引列表
def load_corpus_time_machine(max_tokens=-1):  # @save
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')  # 使用字符级词元化
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]  # 展平成一个列表

    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


if __name__ == '__main__':
    lines = read_time_machine()
    print(f'# 文本总行数: {len(lines)}')
    print(lines[0])
    print(lines[10])

    tokens = tokenize(lines)
    for i in range(11):
        print(tokens[i])

    vocab = Vocab(tokens)
    print(list(vocab.token_to_idx.items())[:10])

    for i in [0, 10]:
        print('文本:', tokens[i])
        print('索引:', vocab[tokens[i]])

    corpus, vocab = load_corpus_time_machine()
    print(f'corpus 长度: {len(corpus)}, vocab 大小: {len(vocab)}')
