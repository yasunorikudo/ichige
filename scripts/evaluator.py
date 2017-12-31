#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 Yasunori Kudo

import chainer
from chainer import cuda
from chainer import function
import chainer.functions as F
from chainer import reporter as reporter_module
from chainer.training import extensions

import copy
import numpy as np
import os


class Evaluator(extensions.Evaluator):

    def __init__(self, dataset, updater, out,
                 synsets, f_list, *args, **kwargs):
        super(Evaluator, self).__init__(*args, **kwargs)
        self.dataset = dataset
        self.updater = updater
        self.out = out
        self.synsets = synsets
        self.f_list = f_list

    def evaluate(self):
        iterator = self._iterators['main']
        model = self._targets['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        mats = None
        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                x, t = self.converter(batch, self.device)
                with function.no_backprop_mode():
                    y = model.predictor(x)
                    loss = F.softmax_cross_entropy(y, t)
                    acc = F.accuracy(y, t)
                    chainer.report({'loss': loss, 'accuracy': acc}, model)

                    mat = cuda.to_cpu(F.softmax(y).data)
                    if mats is None:
                        mats = mat.copy()
                    else:
                        mats = np.concatenate((mats, mat), axis=0)

            summary.add(observation)

        # Save prediction result.
        with open(os.path.join(self.out, 'prediction_epoch_{}.csv'.format(
                self.updater.epoch)), 'w') as f:
            f.write('ID,Image Name,Ground Truth')
            for i, word in enumerate(self.synsets):
                f.write(',{} {}'.format(i, word))
            f.write('\n')
            for one_batch, mat in zip(self.dataset.batch_list, mats):
                features_id = one_batch['features_id']
                start_pos = one_batch['start_pos']
                l_sequence = one_batch['l_sequence']
                middle_pos = start_pos + (l_sequence - 1) // 2
                labels = self.dataset.labels_all[features_id]
                img_name = list(labels.keys())[middle_pos]
                f.write('{},{},{}'.format(
                    self.f_list[features_id], img_name, labels[img_name]))
                for score in mat:
                    f.write(',{}'.format(score))
                f.write('\n')

        return summary.compute_mean()
