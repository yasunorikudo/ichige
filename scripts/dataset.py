#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 Yasunori Kudo

import numpy as np
import chainer
import json


class Dataset(chainer.dataset.DatasetMixin):

    def __init__(self, features_list, labels_list, l_sequence=151, train=False):
        self.features_list = features_list
        self.labels_list = labels_list
        self.l_sequence = l_sequence
        self.train = train

        # Load features and labels.
        self.features_all = []
        self.labels_all = []
        for features_path, labels_path in zip(features_list, labels_list):
            # Load features.
            with open(features_path) as f:
                features = json.load(f)
            self.features_all.append(features)

            # Load labels.
            labels = {}
            with open(labels_path) as f:
                f.readline()  # Skip first row.
                for line in f:
                    img_name, t, _ = line.strip().split(',')
                    t = int(t)
                    labels[img_name] = t
            self.labels_all.append(labels)

        # Create mini-bathc list.
        self.batch_list = []
        self.use_keys = ['dx', 'dy']
        for j, features in enumerate(self.features_all):
            features_keys = list(features.keys())
            N = len(features_keys)
            for start_pos in range(0, N - self.l_sequence + 1):
                # Check if the sequence is valid.
                valid = True
                for i in range(self.l_sequence):
                    img_name = features_keys[start_pos + i]
                    if features[img_name] is None:
                        valid = False
                        break
                    if valid:
                        for use_key in self.use_keys:
                            if not use_key in list(features[img_name].keys()):
                                valid = False
                    if not valid:
                        break

                # Add the sequence to mini-batch list if the sequence is valid.
                if valid:
                    self.batch_list.append({
                        'features_id': j,
                        'start_pos': start_pos,
                        'l_sequence': self.l_sequence
                    })

    def __len__(self):
        return len(self.batch_list)

    def get_example(self, i):
        features_id = self.batch_list[i]['features_id']
        start_pos = self.batch_list[i]['start_pos']
        l_sequence = self.batch_list[i]['l_sequence']

        features = self.features_all[features_id]
        img_names = list(features.keys())

        x = []
        for j in range(start_pos, start_pos + l_sequence):
            img_name = img_names[j]
            for use_key in self.use_keys:
                x.append(features[img_name][use_key])
        x = np.asarray(x, dtype=np.float32)

        if self.train:
            if int(np.random.randint(0, 2, 1)) == 0:
                x[0::2] *= -1  # Randomly flip dx.
            if int(np.random.randint(0, 2, 1)) == 0:
                x[1::2] *= -1  # Randomly flip dy.

        middle_pos = start_pos + (l_sequence - 1) // 2
        img_name = img_names[middle_pos]
        labels = self.labels_all[features_id]
        t = labels[img_name]
        t = np.asarray(t, dtype=np.int32)

        return x, t
