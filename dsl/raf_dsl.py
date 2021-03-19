"""
MIT License

Copyright (c) 2019 Soham Pal, Yash Gupta, Aditya Shukla, Aditya Kanade,
Shirish Shevade, Vinod Ganapathy. Indian Institute of Science.

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os
import cv2
from cfg import cfg
import numpy as np
from dsl.base_dsl import BaseDSL, one_hot_labels


class RAFDSL(BaseDSL):
    def __init__(self, batch_size, shuffle_each_epoch=False, seed=1337, normalize=True, mode='train', val_frac=0.02,
                 normalize_channels=False, resize=None):
        assert mode == "train" or mode == "val" or mode == "test"

        self.shape = (cfg.img_size, cfg.img_size, 3)
        self.mode = mode
        self.normalize = normalize

        if mode == 'val':
            assert val_frac is not None

        super(RAFDSL, self).__init__(
            batch_size,
            shuffle_each_epoch=shuffle_each_epoch,
            seed=seed,
            normalize=False,
            mode=mode,
            val_frac=val_frac,
            normalize_channels=False,
            resize=resize
        )

    def is_multilabel(self):
        return False

    def load_variable(self, file_path, data_type, var_shape):
        var = np.fromfile(file_path, dtype=data_type)
        var.shape = var_shape
        return var

    def get_sample_shape(self):
        return self.shape

    def create_label_dict(self):
        label_dict = {}
        for (img_name, pred_label) in zip(self.data, self.pred_labels):
            label_dict[img_name] = pred_label

        return label_dict

    def create_pred_vector_dict(self):
        pred_vector_dict = {}
        for (img_name, pred_vector) in zip(self.data, self.pred_vectors):
            pred_vector_dict[img_name] = pred_vector

        return pred_vector_dict

    def load_data(self, mode, val_frac):
        self.train_set_size = 11920
        self.test_set_size = 2973

        if mode == 'test':
            # self.test_size = self.test_set_size
            self.test_size = 1000
            # self.labels = self.load_variable(file_path="raf/test_set_true_labels.bin",
            #                                  data_type="int64",
            #                                  var_shape=(test_set_size,))
            self.labels = self.load_variable(file_path="raf/test_set_pred_labels.bin",
                                             data_type="int64",
                                             var_shape=(self.test_set_size,))
            self.pred_labels = self.load_variable(file_path="raf/test_set_pred_labels.bin",
                                                  data_type="int64",
                                                  var_shape=(self.test_set_size, 1))
            self.data = self.load_variable(file_path="raf/test_set_img_names.bin",
                                           data_type="<U13",
                                           var_shape=(self.test_set_size,))
            self.pred_vectors = self.load_variable(file_path="raf/test_set_pred_vectors.bin",
                                                   data_type="float32",
                                                   var_shape=(self.test_set_size, 7))

            self.labels = self.labels[: self.test_size]
            self.data = self.data[: self.test_size]
            self.pred_labels = self.pred_labels[: self.test_size]
            self.pred_vectors = self.pred_vectors[: self.test_size]
        else:
            assert mode == 'train' or mode == 'val'
            # self.train_size = self.train_set_size
            self.train_size = 10000
            # self.labels = self.load_variable(file_path="raf/train_set_true_labels.bin",
            #                                  data_type="int64",
            #                                  var_shape=(train_set_size,))
            self.labels = self.load_variable(file_path="raf/train_set_pred_labels.bin",
                                             data_type="int64",
                                             var_shape=(self.train_set_size,))
            self.pred_labels = self.load_variable(file_path="raf/train_set_pred_labels.bin",
                                                  data_type="int64",
                                                  var_shape=(self.train_set_size, 1))
            self.data = self.load_variable(file_path="raf/train_set_img_names.bin",
                                           data_type="<U15",
                                           var_shape=(self.train_set_size,))
            self.pred_vectors = self.load_variable(file_path="raf/train_set_pred_vectors.bin",
                                                   data_type="float32",
                                                   var_shape=(self.train_set_size, 7))

            self.labels = self.labels[: self.train_size]
            self.data = self.data[: self.train_size]
            self.pred_labels = self.pred_labels[: self.train_size]
            self.pred_vectors = self.pred_vectors[: self.train_size]

        self.label_dict = self.create_label_dict()
        self.pred_vector_dict = self.create_pred_vector_dict()


        # Perform splitting
        if val_frac is not None:
            self.partition_validation_set(mode, val_frac)

        self.labels = np.squeeze(self.labels)

    def convert_Y(self, Y):
        return one_hot_labels(Y, 7)
