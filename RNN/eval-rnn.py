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
import math
demb = 100


def cal_ps(model, s):
    h = Variable(np.zeros((1, demb), dtype=np.float32))
    sum = 0.0
    for i in range(1, len(s)):
        w1, w2 = s[i-1], s[i]
        x_k = model.embed(Variable(np.array([w1], dtype=np.int32)))
        h = F.tanh(x_k + model.H(h))
        yv = F.softmax(model.W(h))
        pi = yv.data[0][w2]
        sum -= math.log(pi, 2)
    return sum


def main():
    test_data = load_data('ptb.test.txt')
    test_data = test_data[0:1000]

if __name__ == '__main__':
    main()
