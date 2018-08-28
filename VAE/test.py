#!/usr/bin/env python
import argparse
import os

import chainer
from chainer import training
from chainer.training import extensions
from chainer import serializers
from chainer import optimizers
from chainer import iterators
from chainer import Variable
import numpy as np

import net


def main():
    parser = argparse.ArgumentParser(description='Chainer test:')
    parser.add_argument('--initmodel', '-m', default='',
                        help='Initialize the model from given file')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the optimization from snapshot')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--epoch', '-e', default=100, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--dimz', '-z', default=20, type=int,
                        help='dimention of encoded vector')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='learning minibatch size')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.add_argument('--vae', action='store_true',
                        help='pre-training vae')
    args = parser.parse_args()

    train, test = chainer.datasets.get_mnist(ndim=3)
    print('before:train: {0}, test: {1}'.format(len(train), len(test)))
    if args.vae:
        data = [[i]*2 for i in range(len(train))]
        for i, d in enumerate(train):
            data[i][1] = train[i][1]
        model = net.VAE(784, args.dimz, 500)
        serializers.load_npz('model/vae.npz', model)
        for i, x in enumerate(train):
            x1 = model.forward(x)
            data[i][0] = np.array(x1, dtype=float32)

    print('after:train: {0}, test: {1}'.format(len(train), len(test)))
    sys.exit()
    # モデルの生成
    model = net.CNN()
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
        x = Variable(np.array([test[i][0]], dtype=np.float32))
        t = test[i][1]
        out = model.fwd(x)
        ans = np.argmax(out.data)
        if (ans == t):
            ok += 1

    print((ok * 1.0) / len(test))


if __name__ == '__main__':
    main()
