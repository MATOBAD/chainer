#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import Variable
import chainer.links as L
import numpy as np


class BLSTMBase(Chain):
    def __init__(self, embeddings, n_labels, dropout=0.5, train=True):
        vocab_size, embed_size = embeddings.shape
        feature_size = embed_size
        super(BLSTMBase, self).__init__(
            embed=L.EmbedID(
                in_size=vocab_size,
                out_size=embed_size,
                initialW=embeddings,
            ),
            f_lstm=L.LSTM(feature_size, feature_size, dropout),
            b_lstm=L.LSTM(feature_size, feature_size, dropout),
            linear=L.Linear(feature_size * 2, n_labels),
        )
        self._dropout = dropout
        self._n_labels = n_labels
        self.train = train

    def reset_state(self):
        self.f_lstm.reset_state()
        self.b_lstm.reset_state()

    def __call__(self, xs):
        self.reset_state()
        xs_f = []
        xs_b = []
        for x in xs:
            _x = self.embed(self.xp.array(x))
            xs_f.append(_x)
            xs_b.append(_x[::-1])
        hs_f = self.f_lstm(xs_f, self.train)
        hs_b = self.b_lstm(xs_b, self.train)
        ys = [self.linear(
              F.dropout(F.concat([h_f, h_b[::-1]]),
                        ratio=self._dropout,
                        train=self.train)) for h_f, h_b in zip(hs_f, hs_b)]
        return ys
