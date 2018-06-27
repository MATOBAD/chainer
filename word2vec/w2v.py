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
import collections
from chainer.utils import walker_alias  # 離散型の確率分布からの乱数生成アルゴリズム


class MyW2V(Chain):
    def __init__(self, n_vocab, n_units):
        super(MyW2V, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
        )

    def __call__(self, xb, yb, tb):
        # 単語のidのペア(xbとyb)と教師信号tb
        xc = Variable(np.array(xb, dtype=np.int32))
        yc = Variable(np.array(yb, dtype=np.int32))
        tc = Variable(np.array(tb, dtype=np.int32))
        fv = self.fwd(xc, yc)
        return F.softmax_cross_entropy(fv, tc)

    def fwd(self, x):
        xv = self.embed(x)  # 単語id xに対する分散表現
        yv = self.embed(y)  # 単語id yに対する分散表現
        return F.sum(xv * yv, axis=1)


if __name__ == "__main__":
    index2word = {}  # 単語のid番号から単語を取り出す辞書
    word2index = {}  # 単語から単語のid番号を取り出す辞書
    couunts = collections.Counter()
    dataset = []  # 単語のid番号のリスト
    with open('ptb.train.txt') as f:
        for line in f:
            for word in line.split():
                if word not in word2index:
                    ind = len(word2index)
                    word2index[word] = ind
                    index2word[ind] = word
                couunts[word2index[word]] += 1
                dataset.append(word2index[word])
    n_vocab = len(word2index)
    datasize = len(dataset)
    cs = [couunts[w] for w in range(len(couunts))]
    power = np.float32(0.75)
    p = np.array(cs, power.dtype)
    sampler = walker_alias.WalkerAlias(p)

    # モデルの生成
    model = MyModel()
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    # パラメータの更新
    iterator = iterators.SerialIterator(train, 1000)
    updater = training.StandardUpdater(iterator, optimizer)
    trainer = training.Trainer(updater, (10, 'epoch'))

    trainer.run()

    # 評価
    ok = 0
    for i in range(len(test)):
        x = Variable(np.array([test[i][0]], dtype=np.float32))
        t = test[i][1]
        out = model.fwd(x)
        ans = np.argmax(out.data)
        if (ans == t):
            ok += 1

    print((ok * 1.0) / len(test))
