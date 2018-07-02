#!/usr/bin/env python
# coding: utf-8
import numpy as np
import argparse
import chainer
from chainer import cuda, Function, \
    report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extension
from chainer.datasets import tuple_dataset
from dataset import mnist


class MyModel(Chain):
    def __init__(self):
        super(MyModel, self).__init__(
            cn1=L.Convolution2D(1, 20, 5),
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


def main():
    xp = cuda.cupy
    # モデルの生成
    model = MyModel()
    if args.gpu >= 0:
        (x_train, t_train), (x_test, t_test) =\
            load_mnist(normalize=True, one_hot_label=True)
        train = tuple(x_train, t_train)
        test = tuple(x_test, t_test)
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU
    else:
        train, test = datasets.get_mnist(ndim=3)
        xp = np
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    train, test = datasets.get_mnist(ndim=3, dtype=xp.float32)

    # パラメータの更新
    iterator = iterators.SerialIterator(train, 1000)
    updater = training.StandardUpdater(iterator, optimizer, device=args.gpu)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()
    main()
