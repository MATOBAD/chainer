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
from chainer.training import extension
from sklearn import datasets


class IrisChain(Chain):
    def __init__(self):
        super(IrisChain, self).__init__(
            l1=L.Linear(4, 6),
            l2=L.Linear(6, 3),
        )

    def __call__(self, x, y):
        return F.softmax_cross_entropy(self.fwd(x), y)

    def fwd(self, x):
        h1 = F.sigmoid(self.l1(x))
        h2 = self.l2(h1)
        return h2


if __name__ == "__main__":
    # irisのデータセット
    iris = datasets.load_iris()
    X = iris.data.astype(np.float32)
    Y = iris.target.astype(np.int32)
    N = Y.size
    index = np.arange(N)
    xtrain = X[index[index % 2 != 0], :]
    ytrain = Y[index[index % 2 != 0]]  # 教師信号は整数値
    xtest = X[index[index % 2 == 0], :]
    yans = Y[index[index % 2 == 0]]

    # モデルの生成
    model = IrisChain()
    optimizer = optimizers.SGD()
    optimizer.setup(model)

    # パラメータの更新(ミニバッチ)
    n = 75  # データのサイズ
    bs = 25  # バッチサイズ
    for i in range(5000):
        sffindx = np.random.permutation(n)
        for j in range(0, n, bs):
            x = Variable(xtrain[sffindx[j:(j+bs) if (j+bs) < n else n]])
            y = Variable(ytrain[sffindx[j:(j+bs) if (j+bs) < n else n]])
            model.cleargrads()
            loss = model(x, y)
            loss.backward()
            optimizer.update()

    # モデルの評価
    xt = Variable(xtest)
    yt = model.fwd(xt)
    ans = yt.data
    nrow, ncol = ans.shape
    ok = 0
    for i in range(nrow):
        cls = np.argmax(ans[i, :])
        if cls == yans[i]:
            ok += 1

    # 結果の出力
    print("{0}/{1}={2}".format(ok, nrow, (ok * 1.0)/nrow))
