import os
import time

import cv2
import torch
import numpy as np

from load_kdef_torch_2_bin import save_variable
from load_kdef_torch_2_bin import load_variable
from load_kdef_torch_2_bin import calculate_acc

RAF_DIR = "raf"
ORIGINAL_RAF_DIR = "raf/original"

MAX_HEIGHT = 3221
MAX_WIDTH = 2147


def label_emotion_in_api_index(raf_labels):
    for (img_name, raf_label) in raf_labels.items():
        if raf_label == "6":  # anger
            raf_labels[img_name] = 0
        elif raf_label == "3":  # disgust
            raf_labels[img_name] = 1
        elif raf_label == "2":  # fear
            raf_labels[img_name] = 2
        elif raf_label == "4":  # happiness
            raf_labels[img_name] = 3
        elif raf_label == "7":  # neutral
            raf_labels[img_name] = 4
        elif raf_label == "5":  # sadness
            raf_labels[img_name] = 5
        elif raf_label == "1":  # surprise
            raf_labels[img_name] = 6
        else:
            raise Exception("No such emotion in api: {}-{}".format(img_name, raf_label))


def load_raf_api_results(api_results, api_true_labels_dict):
    train_set_img_names = []
    train_set_true_labels = []
    train_set_pred_labels = []
    train_set_pred_vectors = None

    test_set_img_names = []
    test_set_true_labels = []
    test_set_pred_labels = []
    test_set_pred_vectors = None

    for (key, value) in api_results.items():
        img_name = key.split("/")[-1]
        pred_vector = value.numpy()
        pred_label = np.where(pred_vector == np.max(pred_vector))[1]

        data_set_type = img_name.split("_")[0]
        print("load {} - {}".format(data_set_type, img_name))
        if data_set_type == "train":
            train_set_img_names.append(img_name)
            train_set_true_labels.append(api_true_labels_dict[img_name])
            train_set_pred_labels.append(pred_label)

            if train_set_pred_vectors is None:
                train_set_pred_vectors = pred_vector
            else:
                train_set_pred_vectors = np.vstack((train_set_pred_vectors, pred_vector))

        elif data_set_type == "test":
            test_set_img_names.append(img_name)
            test_set_true_labels.append(api_true_labels_dict[img_name])
            test_set_pred_labels.append(pred_label)

            if test_set_pred_vectors is None:
                test_set_pred_vectors = pred_vector
            else:
                test_set_pred_vectors = np.vstack((test_set_pred_vectors, pred_vector))

        else:
            raise Exception("No such data set type: {}, {}".format(data_set_type, key))

    # train_set_data
    train_set_img_names = np.array(train_set_img_names)
    train_set_true_labels = np.array(train_set_true_labels)
    train_set_pred_labels = np.array(train_set_pred_labels)

    # test_set_data
    test_set_img_names = np.array(test_set_img_names)
    test_set_true_labels = np.array(test_set_true_labels)
    test_set_pred_labels = np.array(test_set_pred_labels)

    return (train_set_img_names, train_set_true_labels, train_set_pred_labels, train_set_pred_vectors), \
           (test_set_img_names, test_set_true_labels, test_set_pred_labels, test_set_pred_vectors)


def load_img(img_name, normalize=True):
    img_path = os.path.join("raf/original", img_name)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (32, 32))
    if normalize:
        img = np.divide(img, 255).astype("float16")

    return img

def normalize_img(img_name):
    normalized_img = load_img(img_name, normalize=True)
    img_name_no_suffix = img_name.split(".")[0]
    if not os.path.exists("raf/normalized_imgs"):
        os.mkdir("raf/normalized_imgs")
    save_variable(normalized_img, img_name_no_suffix, data_dir="raf/normalized_imgs")


def normalize_imgs(api_results):
    outset = time.time()
    for img_path in api_results.keys():
        img_name = img_path.split("/")[-1]
        normalize_img(img_name)
    print("took {} min".format(round((time.time() - outset) / 60, 2)))

if __name__ == "__main__":
    raf_api_results = torch.load("raf/output_tensor_dict4.pt")
    raf_true_labels = torch.load("raf/dic_partition_label.pt")

    normalize_imgs(raf_api_results)

    label_emotion_in_api_index(raf_true_labels)

    (train_set_img_names, train_set_true_labels,
     train_set_pred_labels, train_set_pred_vectors), \
    (test_set_img_names, test_set_true_labels,
     test_set_pred_labels, test_set_pred_vectors) = load_raf_api_results(raf_api_results, raf_true_labels)

    train_set_acc = calculate_acc(train_set_true_labels, train_set_pred_labels)
    test_set_acc = calculate_acc(test_set_true_labels, test_set_pred_labels)

    save_variable(train_set_img_names, "train_set_img_names", data_dir="raf")
    save_variable(train_set_true_labels, "train_set_true_labels", data_dir="raf")
    save_variable(train_set_pred_labels, "train_set_pred_labels", data_dir="raf")
    save_variable(train_set_pred_vectors, "train_set_pred_vectors", data_dir="raf")
    save_variable(train_set_acc, "train_set_acc", data_dir="raf")

    save_variable(test_set_img_names, "test_set_img_names", data_dir="raf")
    save_variable(test_set_true_labels, "test_set_true_labels", data_dir="raf")
    save_variable(test_set_pred_labels, "test_set_pred_labels", data_dir="raf")
    save_variable(test_set_pred_vectors, "test_set_pred_vectors", data_dir="raf")
    save_variable(test_set_acc, "test_set_acc", data_dir="raf")

    train_set_img_names = load_variable(file_path="raf/train_set_img_names.bin",
                                        data_type="<U15",
                                        var_shape=(11920,))
    train_set_true_labels = load_variable(file_path="raf/train_set_true_labels.bin",
                                          data_type="int64",
                                          var_shape=(11920, 1))
    train_set_pred_labels = load_variable(file_path="raf/train_set_pred_labels.bin",
                                          data_type="int64",
                                          var_shape=(11920, 1))
    train_set_pred_vectors = load_variable(file_path="raf/train_set_pred_vectors.bin",
                                           data_type="float32",
                                           var_shape=(11920, 7))
    train_set_acc = load_variable(file_path="raf/train_set_acc.bin",
                                  data_type="float64",
                                  var_shape=1)[0]

    test_set_img_names = load_variable(file_path="raf/test_set_img_names.bin",
                                       data_type="<U13",
                                       var_shape=(2973,))
    test_set_true_labels = load_variable(file_path="raf/test_set_true_labels.bin",
                                       data_type="int64",
                                       var_shape=(2973, 1))
    test_set_pred_labels = load_variable(file_path="raf/test_set_pred_labels.bin",
                                       data_type="int64",
                                       var_shape=(2973, 1))
    test_set_pred_vectors = load_variable(file_path="raf/test_set_pred_vectors.bin",
                                       data_type="float32",
                                       var_shape=(2973, 7))
    test_set_acc = load_variable(file_path="raf/test_set_acc.bin",
                                       data_type="float64",
                                       var_shape=1)[0]
