from __future__ import absolute_import
from __future__ import division
from utils.model_v2 import *
from utils.class_loader import *
from cfg import cfg, config
import tensorflow as tf

assert cfg.copy_model == 'vgg16' and cfg.img_size == 224 and cfg.noise_dataset == 'nsfw'

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

dataset_dsl = load_dataset(cfg.noise_dataset)

train_dsl = dataset_dsl(batch_size = cfg.batch_size, mode='train', shuffle_each_epoch=False, seed=cfg.seed)
val_dsl   = dataset_dsl(batch_size = cfg.batch_size, mode='val', shuffle_each_epoch=False, seed=cfg.seed)
test_dsl  = dataset_dsl(batch_size = cfg.batch_size, mode='test', shuffle_each_epoch=False, seed=cfg.seed)

tf.set_random_seed(cfg.seed)

with tf.Session(config=config) as sess:
    from tensorflow_vgg16.vgg16 import Vgg16Wrapper
    copy_model = Vgg16Wrapper()
    sess.run(tf.global_variables_initializer())
    copy_model.load_weights(sess)

    train_copynet_iter(copy_model, train_dsl, val_dsl, test_dsl, sess)

"""
To run this program in the background:
$ nohup python generic_program_v2.py > nsfw_vgg16_log.output 2>&1 &
"""

