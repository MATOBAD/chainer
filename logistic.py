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


class IrisLogi(Chain):
    def __init__(self):
        super(IrisLogi, self).__init__(
            l1=L.Linear(4, 3),
        )

    def __call__(self, x, y):
        return F.mean_squared_error(self.fwd(x), y)

    def fwd(self, x):
        return F.softmax(self.l1(x))


if __name__ == "__main__":
    # irisのデータセット
    iris = datasets.load_iris()
    X = iris.data.astype(np.float32)
    Y = iris.target
    N = Y.size
    Y2 = np.zeros(3 * N).reshape(N, 3).astype(np.float32)
    for i in range(N):
        Y2[i, Y[i]] = 1.0
    index = np.arange(N)
    xtrain = X[index[index % 2 != 0], :]
    ytrain = Y2[index[index % 2 != 0], :]
    xtest = X[index[index % 2 == 0], :]
    yans = Y[index[index % 2 == 0]]

    # モデルの生成
    model = IrisLogi()
    optimizer = optimizers.Adam()
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
