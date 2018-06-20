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

if __name__ == "__main__":
    x1 = Variable(np.array([-1], dtype=np.float))
    print("sin(x1):", F.sin(x1).data)  # sin関数
    print("sigmoid(x1):", F.sigmoid(x1).data)  # シグモイド関数

    x2 = Variable(np.array([-0.5], dtype=np.float))
    z1 = F.cos(x2)
    print("cos(x2):", z1.data)
    z1.backward()
    print("微分:", x2.grad)
    print("確認:", ((-1) * F.sin(x2)).data)

    x3 = Variable(np.array([-1, 0, 1], dtype=np.float))
    z2 = F.sin(x3)
    z2.grad = np.ones(3, dtype=np.float)
    z2.backward()
    print("3次元の微分:", x3.grad)
