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
ws = 3  # window size
ngs = 5  # negative sample size


# 単語のペアの集合のバッチを作る関数
def mkbatset(dataset, ids):
    xb, yb, tb = [], [], []
    for pos in ids:
        xid = dataset[pos]
        for i in range(1, ws):
            p = pos - i
            if p >= 0:
                xb.append(xid)
                yid = dataset[p]
                yb.append(yid)
                tb.append(1)
                for nid in sampler.sample(ngs):
                    xb.append(yid)
                    yb.append(nid)
                    tb.append(0)
            p = pos + i
            if p < datasize:
                xb.append(xid)
                yid = dataset[p]
                yb.append(yid)
                tb.append(1)
                for nid in sampler.sample(ngs):
                    xb.append(yid)
                    yb.append(nid)
                    tb.append(0)
    return [xb, yb, tb]


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
        return F.sigmoid_cross_entropy(fv, tc)

    def fwd(self, x, y):
        xv = self.embed(x)  # 単語id xに対する分散表現
        yv = self.embed(y)  # 単語id yに対する分散表現
        return F.sum(xv * yv, axis=1)


if __name__ == "__main__":
    # 単語辞書の作成
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
    # サンプル生成器(離散型の確率分布)
    sampler = walker_alias.WalkerAlias(p)

    # モデルの生成
    demb = 100  # 分散表現の次元数
    model = MyW2V(n_vocab, demb)
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    # パラメータの更新
    bs = 100  # batch size
    print(datasize)
    for epoch in range(10):
        print('epoch: {0}'.format(epoch))
        indexes = np.random.permutation(datasize)
        for pos in range(0, datasize, bs):
            print(epoch, pos)
            ids = indexes[pos:(pos+bs) if (pos+bs) < datasize else datasize]
            xb, yb, tb = mkbatset(dataset, ids)
            model.cleargrads()
            loss = model(xb, yb, tb)
            loss.backward()
            optimizer.update()

    with open('myw2v.model', 'w') as f:
        f.write('%d %d\n' % (len(index2word), 100))
        w = model.embed.W.data
        for i in range(w.shape[0]):
            v = ' '.join(['%f' % v for v in w[i]])
            f.write('%s %s\n' % (index2word[i], v))
