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

import numpy as np
from dsl.base_dsl import BaseDSL, one_hot_labels


class KDEFDSL(BaseDSL):
    def __init__(self, batch_size, shuffle_each_epoch=False, seed=1337, normalize=True, mode='train', val_frac=0.2,
                 normalize_channels=False, path=None, resize=None):
        assert mode == "train" or mode == "val" or mode == "test"

        self.data_set_type = "kdef"
        self.orig_dsl = self
        self.shape = (32, 32, 3)
        self.mode = mode
        self.normalize = normalize

        if mode == 'val':
            assert val_frac is not None

        super(KDEFDSL, self).__init__(
            batch_size,
            shuffle_each_epoch=shuffle_each_epoch,
            seed=seed,
            normalize=False,
            mode=mode,
            val_frac=val_frac,
            normalize_channels=normalize_channels,
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

    def load_data(self, mode, val_frac):
        train_set_size = 4076
        test_set_size = 815

        if mode == 'test':
            test_size = test_set_size  # 808
            # self.labels = self.load_variable(file_path="kdef/test_set_true_labels.bin",
            #                                  data_type="int32",
            #                                  var_shape=(test_set_size,))
            # self.labels = self.labels.astype("int64")
            self.labels = self.load_variable(file_path="kdef/test_set_pred_labels.bin",
                                             data_type="int64",
                                             var_shape=(test_set_size, 1))
            self.pred_labels = self.load_variable(file_path="kdef/test_set_pred_labels.bin",
                                                  data_type="int64",
                                                  var_shape=(test_set_size, 1))
            self.data = self.load_variable(file_path="kdef/test_set_filenames.bin",
                                           data_type="<U12",
                                           var_shape=(test_set_size,))
            self.pred_vectors = self.load_variable(file_path="kdef/test_set_pred_vectors.bin",
                                                   data_type="float32",
                                                   var_shape=(test_set_size, 7))

            self.labels = self.labels[: test_size]
            self.data = self.data[: test_size]
            self.pred_labels = self.pred_labels[: test_size]
            self.pred_vectors = self.pred_vectors[: test_size]
        else:
            assert mode == 'train' or mode == 'val'
            train_size = train_set_size  # 4076
            # self.labels = self.load_variable(file_path="kdef/train_set_true_labels.bin",
            #                                  data_type="int32",
            #                                  var_shape=(train_set_size,))
            # self.labels = self.labels.astype("int64")
            self.labels = self.load_variable(file_path="kdef/train_set_pred_labels.bin",
                                             data_type="int64",
                                             var_shape=(train_set_size, 1))
            self.pred_labels = self.load_variable(file_path="kdef/train_set_pred_labels.bin",
                                                  data_type="int64",
                                                  var_shape=(train_set_size, 1))
            self.data = self.load_variable(file_path="kdef/train_set_filenames.bin",
                                           data_type="<U12",
                                           var_shape=(train_set_size,))
            self.pred_vectors = self.load_variable(file_path="kdef/train_set_pred_vectors.bin",
                                                   data_type="float32",
                                                   var_shape=(train_set_size, 7))

            self.labels = self.labels[: train_size]
            self.data = self.data[: train_size]
            self.pred_labels = self.pred_labels[: train_size]
            self.pred_vectors = self.pred_vectors[: train_size]

        # Perform splitting
        if val_frac is not None:
            self.partition_validation_set(mode, val_frac)

        self.labels = np.squeeze(self.labels)

    def convert_Y(self, Y):
        return one_hot_labels(Y, 7)
