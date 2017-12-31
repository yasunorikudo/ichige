#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 Yasunori Kudo

import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

import dataset
import evaluator

class MLP(chainer.Chain):

    def __init__(self, n_class=8):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, 512)
            self.l2 = L.Linear(512, 512)
            self.l3 = L.Linear(512, n_class)

    def __call__(self, x):
        h = F.tanh(self.l1(x))
        h = F.tanh(self.l2(h))
        return self.l3(h)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=30,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--train_set', '-t', type=str, nargs='*', default=[1, 2, 3],
                        help='List of index of training set')
    parser.add_argument('--val_set', '-v', type=str, nargs='*', default=[4],
                        help='List of index of validation set')
    parser.add_argument('--synsets', '-s', type=str, default='data/label1.csv',
                        help='Path to label name file')
    parser.add_argument('--l_sequence', '-l', type=int, default=151,
                        help='Length of sequences')
    args = parser.parse_args()

    model = L.Classifier(MLP())
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
    optimizer = chainer.optimizers.NesterovAG(lr=3e-3)
    optimizer.setup(model)

    train_features_list = [
        'data/features{}.json'.format(i) for i in args.train_set]
    val_features_list = [
        'data/features{}.json'.format(i) for i in args.val_set]
    train_labels_list = [
        'data/annotation{}.csv'.format(i) for i in args.train_set]
    val_labels_list = [
        'data/annotation{}.csv'.format(i) for i in args.val_set]
    train = dataset.Dataset(train_features_list, train_labels_list,
                            l_sequence=args.l_sequence, train=True)
    val = dataset.Dataset(val_features_list, val_labels_list,
                          l_sequence=args.l_sequence, train=False)
    train_iter = chainer.iterators.SerialIterator(
        train, batch_size=args.batchsize, shuffle=True, repeat=True)
    val_iter = chainer.iterators.SerialIterator(
        val, batch_size=args.batchsize, shuffle=False, repeat=False)

    updater = training.updater.StandardUpdater(
        train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    synsets = []
    with open(args.synsets) as f:
        for line in f:
            synsets.append(line.strip().split(',')[1])
    trainer.extend(evaluator.Evaluator(
        dataset=val, iterator=val_iter, target=model, updater=updater,
        device=args.gpu, out=args.out, synsets=synsets,
        f_list=val_features_list))
    trainer.extend(extensions.snapshot_object(
        model, 'model_epoch_{.updater.epoch}.npz'), trigger=(1, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    trainer.run()
