#!/usr/bin/env python
# coding: utf-8
import numpy as np
import chainer
from chainer import cuda, Function, \
    report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
vocab = {}  # 単語辞書のid


def load_data(filename):
    global vocab
    words = open(filename).read().replace('\n', '<eos>').strip().split()
    # <eos>: 文の終わりを表す記号
    dataset = np.ndarray((len(words), ), dtype=np.int32)  # 単語数
    for i, word in enumerate(words):
        if word not in vocab:
            vocab[word] = len(vocab)
        dataset[i] = vocab[word]
    return dataset

train_data = load_data('ptb.train.txt')
eos_id = vocab['<eos>']


class MyRNN(Chain):
    def __init__(self, v, k):
        """
        v: ボキャブラリの数
        k: 分散表現の次元数
        """
        super(MyRNN, self).__init__(
            embed=L.EmbedID(v, k),  # 入力(one-hot)
            H=L.Linear(k, k),
            W=L.Linear(k, v),
        )

    def __call__(self, s):
        accum_loss = None
        v, k = self.embed.W.data.shape
        h = Variable(np.zeros((1, k), dtype=np.float32))
        for i in range(len(s)):
            next_w_id = eos_id if (i == len(s) - 1) else s[i+1]
            tx = Variable(np.array([next_w_id], dtype=np.int32))
            x_k = self.embed(Variable(np.array([s[i]], dtype=np.int32)))
            h = F.tanh(x_k + self.H(h))
            loss = F.softmax_cross_entropy(self.W(h), tx)
            accum_loss = loss if accum_loss is None else accum_loss + loss
        return accum_loss


def main():
    # モデルの生成
    demb = 100
    # demb: 分散表現の次元
    model = MyRNN(len(vocab), demb)
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    # パラメータの更新
    for epoch in range(5):
        s = []
        for pos in range(len(train_data)):
            id = train_data[pos]
            s.append(id)
            if id == eos_id:
                model.cleargrads()
                loss = model(s)
                loss.backward()
                optimizer.update()
                s = []
        outfile = "myrnn-" + str(epoch) + ".model"
        serializers.save_npz(outfile, model)

if __name__ == '__main__':
    main()
