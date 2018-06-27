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
from chainer.cuda import cupy


class MyModel(Chain):
    def __init__(self):
        super(MyModel, self).__init__(
            cn1=L.Convolution2D(1, 20, 5).to_gpu,
            cn2=L.Convolution2D(20, 50, 5),
            l1=L.Linear(800, 500),
            l2=L.Linear(500, 10),
        )

    def __call__(self, x, t):
        return F.softmax_cross_entropy(self.fwd(x), t)

    def fwd(self, x):
        h1 = F.max_pooling_2d(F.relu(self.cn1(x)), 2)
        h2 = F.max_pooling_2d(F.relu(self.cn2(h1)), 2)
        h3 = F.dropout(F.relu(self.l1(h2)))
        return self.l2(h3)


if __name__ == "__main__":
    xp = cuda.cupy
    train, test = datasets.get_mnist(ndim=3, dtype=xp.float32)

    # モデルの生成
    model = MyModel()
    cuda.get_device(0).use()
    model.to_gpu()
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
        x = Variable(xp.array([test[i][0]], dtype=xp.float32))
        t = test[i][1]
        out = model.fwd(x)
        ans = np.argmax(out.data)
        if (ans == t):
            ok += 1

    print((ok * 1.0) / len(test))
