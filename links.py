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
    h = L.Linear(3, 4)
    print("重み:\n", h.W.data)
    print("バイアス:", h.b.data)
    x = Variable(np.array(range(6)).astype(np.float32).reshape(2, 3))
    print("入力:\n", x.data)
    y = h(x)
    print("出力:\n", y.data)

    w = h.W.data
    x0 = x.data
    print("確認:\n", x0.dot(w.T) + h.b.data)
