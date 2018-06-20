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


class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__(
            l1=L.Linear(4, 3),
            l2=L.Linear(3, 3)
        )

    def __call__(self, x, y):
        fv = self.fwd(x, y)
        loss = F.mean_squared_error(fv, y)
        return loss

    def fwd(self, x, y):
        return F.sigmoid(self.l1(x))


if __name__ == "__main__":
    model = MyChain()  # モデルの生成
    optimizer = optimizers.SGD()  # 最適化のアルゴリズムの選択
    optimizer.setup(model)  # アルゴリズムにモデルをセット

    model.cleargrads()  # 勾配の初期化
    loss = model(x, y)  # 順方向に計算して誤差を算出
    loss.backward()  # 逆方向の計算、勾配の計算
    optimizer.update()  # パラメータを更新
