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
    x1 = Variable(np.array([1]).astype(np.float))
    x2 = Variable(np.array([2]).astype(np.float))
    x3 = Variable(np.array([3]).astype(np.float))

    # z = f(x1, x2, x3)
    #   = (x1 - 2 * x2 - 1) ^ 2 + (x2 * x3 - 1) ^ 2 + 1
    z = (x1 - 2 * x2 - 1) ** 2 + (x2 - x3 - 1) ** 2 + 1
    print(z.data)

    z.backward()
    print(x1.grad)
    print(x2.grad)
    print(x3.grad)
