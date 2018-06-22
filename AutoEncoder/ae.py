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
from chainer.datasets import tuple_dataset
from sklearn import datasets
import matplotlib.pyplot as plt


class MyAE(Chain):
    def __init__(self):
        super(MyAE, self).__init__(
            l1=L.Linear(4, 2),
            l2=L.Linear(2, 4),
        )

    def __call__(self, x):
        bv = self.fwd(x)
        return F.mean_squared_error(bv, x)

    def fwd(self, x):
        fv = F.sigmoid(self.l1(x))
        bv = self.l2(fv)
        return bv


if __name__ == "__main__":
    iris = datasets.load_iris()
    xtrain = iris.data.astype(np.float32)

    # モデルの生成
    model = MyAE()
    optimizer = optimizers.SGD()
    optimizer.setup(model)

    # パラメータの更新
    n = 150
    bsize = 30
    for i in range(3000):
        sffindx = np.random.permutation(n)
        for j in range(0, n, bsize):
            x = Variable(xtrain[sffindx[j:(j+bsize) if (j+bsize) < n else n]])
            model.cleargrads()
            loss = model(x)
            loss.backward()
            optimizer.update()

    # データのプロット
    x = Variable(xtrain)
    yt = F.sigmoid(model.l1(x))
    ans = yt.data
    ansx1 = ans[0:50, 0]
    ansy1 = ans[0:50, 1]
    ansx2 = ans[50:100, 0]
    ansy2 = ans[50:100, 1]
    ansx3 = ans[100:150, 0]
    ansy3 = ans[100:150, 1]
    plt.scatter(ansx1, ansy1, marker="^")
    plt.scatter(ansx2, ansy2, marker="o")
    plt.scatter(ansx3, ansy3, marker="+")
    plt.show()
