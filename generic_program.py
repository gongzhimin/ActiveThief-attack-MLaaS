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

from __future__ import absolute_import
from __future__ import division
from utils.model import *
from utils.class_loader import *
from cfg import cfg
import tensorflow as tf


logging.basicConfig(format='%(message)s', level=logging.INFO, filename="{}_{}_log.log".format(cfg.noise_dataset, cfg.copy_model))

for key, value in tf.flags.FLAGS.__flags.items():
    try:
        logging.info("{} {}".format(key, value.value))
        print "{} {}".format(key, value.value)
    except AttributeError:
        logging.info("{} {}".format(key, value))
        print "{} {}".format(key, value)

assert cfg.copy_model is not None

if cfg.iterative:
    assert cfg.initial_seed is not None
    assert cfg.val_size is not None
    assert cfg.num_iter is not None
    assert cfg.k is not None


print "seed set is ", cfg.seed

noise_dataset_dsl = load_dataset(cfg.noise_dataset)

count = 1

while True:
    try:
        print "Loading data. Attempt {}".format(count)
        noise_test_dsl = noise_dataset_dsl(batch_size=cfg.batch_size, mode="test")
        noise_train_dsl = noise_dataset_dsl(batch_size=cfg.batch_size, mode='train')
        noise_val_dsl = noise_dataset_dsl(batch_size=cfg.batch_size, mode='val', seed=cfg.seed)
        break
    except MemoryError as e:
        if count == 5:
            raise Exception("Memory error could not be resolved using time delay")
        else:
            print "Loading data failed. Waiting for 5 min.."
            time.sleep(300)
        count = count + 1

print "Training data loaded"

height, width, channels = noise_test_dsl.get_sample_shape()
num_classes = noise_test_dsl.get_num_classes()
is_multilabel = noise_test_dsl.is_multilabel()

if cfg.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(cfg.learning_rate)  # None
elif cfg.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(cfg.learning_rate)  # None
elif cfg.optimizer == 'gradientdescent':
    optimizer = tf.train.GradientDescentOptimizer(cfg.learning_rate)
elif cfg.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(cfg.learning_rate)
else:
    assert cfg.optimizer is None
    optimizer = None

tf.reset_default_graph()

tf.set_random_seed(cfg.seed)

if cfg.copy_model == "vgg19":
    from tensorflow_vgg.my_vgg19 import Vgg19
    copy_model = Vgg19(learning_rate=cfg.learning_rate)
elif cfg.copy_model == "vgg16":
    pass
else:
    copy_model_type = load_model(cfg.copy_model)

    with tf.variable_scope("copy_model"):
        test_var = tf.get_variable('foo', shape=(1, 5))

        copy_model = copy_model_type(batch_size=cfg.batch_size, height=height, width=width, channels=channels,
                                     num_classes=num_classes, multilabel=is_multilabel, fc_layers=[], optimizer=optimizer)
        copy_model.test_var = test_var
        copy_model.print_trainable_parameters()
        copy_model.print_arch()

if cfg.copy_source_model:

    if cfg.iterative:
        logging.info("Copying source model using iterative approach")
        train_copynet_iter(copy_model, noise_train_dsl, noise_val_dsl, noise_test_dsl)
    else:
        raise Exception("not implemented")
