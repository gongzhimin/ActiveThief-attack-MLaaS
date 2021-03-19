import os
import json
import numpy as np
from collections import defaultdict as dd
from cfg import cfg

data_dir = 'nsfw'
normal_dir = os.path.join(data_dir, "normal")
porn_dir = os.path.join(data_dir, "porn")
sexy_dir = os.path.join(data_dir, "sexy")

nswf_dict_filename = "nsfw_dict.json"
nsfw_dict_dir = os.path.join(data_dir, nswf_dict_filename)

classes = ["normal", "porn", "sexy"]
class_to_idx = {
    "normal": 0,
    "porn": 1,
    "sexy": 2
}


def add_img_pair(img_list, nsfw_dict, class_name, class_dir):
    if img_list is None:
        return

    for img in img_list:
        img_path = os.path.join(class_dir, img)
        img_pair = (img_path, class_to_idx[class_name])
        nsfw_dict[class_name].append(img_pair)


def save_img_paths(normal_list=None, porn_list=None, sexy_list=None):
    if not os.path.exists(nsfw_dict_dir):
        nsfw_dict = {
            "normal": [],
            "porn": [],
            "sexy": []
        }
    else:
        with open(nsfw_dict_dir, 'r') as f:
            nsfw_dict = json.load(f)

    add_img_pair(normal_list, nsfw_dict, "normal", normal_dir)
    add_img_pair(porn_list, nsfw_dict, "porn", porn_dir)
    add_img_pair(sexy_list, nsfw_dict, "sexy", sexy_dir)

    with open(nsfw_dict_dir, 'w') as f:
        json.dump(nsfw_dict, f, ensure_ascii=False)


def save_nsfw_dict_json():
    for _, _, normal_list in os.walk("nsfw/normal"):
        save_img_paths(normal_list=normal_list)

    for _, _, porn_list in os.walk("nsfw/porn"):
        save_img_paths(porn_list=porn_list)

    for _, _, sexy_list in os.walk("nsfw/sexy"):
        save_img_paths(sexy_list=sexy_list)


def get_partition_to_idxs(samples, ntest=1000):
    partition_to_idxs = {
        'train': [],
        'test': []
    }

    prev_state = np.random.get_state()
    np.random.seed(cfg.DS_SEED)

    print(len(samples))
    classidx_to_idxs = dd(list)
    for idx, s in enumerate(samples):
        classidx = s[1]
        classidx_to_idxs[classidx].append(idx)

    # Shuffle classidx_to_idx
    for classidx, idxs in classidx_to_idxs.items():
        np.random.shuffle(idxs)

    for classidx, idxs in classidx_to_idxs.items():
        partition_to_idxs['test'] += idxs[:ntest]  # A constant no. kept aside for evaluation
        partition_to_idxs['train'] += idxs[ntest:]  # Train on remaining

    # Revert randomness to original state
    np.random.set_state(prev_state)

    return partition_to_idxs


if __name__ == "__main__":
    save_nsfw_dict_json()
    # with open(nsfw_dict_dir, 'r') as f:
    #     nsfw_dict = json.load(f)
    #
    # samples = nsfw_dict["normal"] + nsfw_dict["porn"] + nsfw_dict["sexy"]
    #
    # partition_to_idxs = get_partition_to_idxs(samples)
    #
    # pruned_idxs = partition_to_idxs['test']
    # # Prune (self.imgs, self.samples to only include examples from the required train/test partition
    # samples = [samples[i] for i in pruned_idxs]
