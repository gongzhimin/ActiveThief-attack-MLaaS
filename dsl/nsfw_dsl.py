import json
import os
import cv2
from cfg import cfg
import numpy as np
from collections import defaultdict as dd
from dsl.base_dsl import BaseDSL, one_hot_labels


class NSFWDSL(BaseDSL):
    def __init__(self, batch_size, shuffle_each_epoch=False, seed=1337,
                 normalize=True, mode='train', val_frac=0.02, resize=None):
        assert mode == "train" or mode == "val" or mode == "test"

        self.shape = (cfg.img_size, cfg.img_size, 3)
        self.ntest = cfg.ntest
        self.mode = mode
        self.normalize = normalize

        if mode == 'val':
            assert val_frac is not None

        super(NSFWDSL, self).__init__(
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

    def get_partition_to_idxs(self, samples):
        partition_to_idxs = {
            'train': [],
            'test': []
        }

        prev_state = np.random.get_state()
        np.random.seed(cfg.DS_SEED)

        classidx_to_idxs = dd(list)
        for idx, s in enumerate(samples):
            classidx = s[1]
            classidx_to_idxs[classidx].append(idx)

        # Shuffle classidx_to_idx
        for classidx, idxs in classidx_to_idxs.items():
            np.random.shuffle(idxs)

        for classidx, idxs in classidx_to_idxs.items():
            partition_to_idxs['test'] += idxs[:self.ntest]     # A constant no. kept aside for evaluation
            partition_to_idxs['train'] += idxs[self.ntest:]    # Train on remaining

        # Revert randomness to original state
        np.random.set_state(prev_state)

        return partition_to_idxs

    def create_label_dict(self):
        label_dict = {}
        for (img_name, pred_label) in zip(self.data, self.labels):
            label_dict[img_name] = pred_label

        return label_dict

    def load_data(self, mode, val_frac):
        with open("nsfw/nsfw_dict.json", 'r') as f:
            nsfw_dict = json.load(f)

        samples = nsfw_dict["normal"] + nsfw_dict["porn"] + nsfw_dict["sexy"]
        partition_to_idxs = self.get_partition_to_idxs(samples)

        if mode == 'test':
            pruned_idxs = partition_to_idxs['test']
        else:
            assert mode == 'train' or mode == 'val'
            pruned_idxs = partition_to_idxs['train']

        samples = [samples[i] for i in pruned_idxs]

        self.data = []
        self.labels = []
        for sample in samples:
            self.data.append(sample[0])
            self.labels.append(sample[1])

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

        self.label_dict = self.create_label_dict()



        # Perform splitting
        if val_frac is not None:
            self.partition_validation_set(mode, val_frac)

        self.labels = np.squeeze(self.labels)

    def convert_Y(self, Y):
        return one_hot_labels(Y, 3)
