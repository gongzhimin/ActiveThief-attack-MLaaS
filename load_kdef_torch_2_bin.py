import os
import time

import cv2
import torch
import numpy as np

HEIGHT = 762
WIDTH = 562

def label_emotion(emotion):
    if emotion == 'AN':  # angry
        label = 0
    elif emotion == 'AF':  # afraid
        label = 2
    elif emotion == 'DI':  # disgust
        label = 1
    elif emotion == 'HA':  # happy
        label = 3
    elif emotion == 'NE':  # neutral
        label = 4
    elif emotion == 'SA':  # sad
        label = 5
    elif emotion == 'SU':  # surprise
        label = 6
    else:
        raise Exception("No such emotion!")
    return label


def load_kdef_api_results(img_paths, api_results):
    counter = 0
    filenames = []
    true_labels = []
    pred_labels = []
    pred_vectors = None
    for img_path in img_paths:
        counter += 1
        print("round: ", counter)
        pred_vector = api_results[img_path].numpy()
        img_name = img_path.split("/")[-1]

        if img_name == "AM31H.JPG":
            img_name = "AM31SUHR.JPG"
        elif img_name == "AF31V.JPG":
            img_name = "AF31SAHL.JPG"

        emotion = img_name[4:6]
        true_label = label_emotion(emotion)
        pred_label = np.where(pred_vector == np.max(pred_vector))[1]

        filenames.append(img_name)
        true_labels.append(true_label)
        pred_labels.append(pred_label)

        if pred_vectors is None:
            pred_vectors = pred_vector
        else:
            pred_vectors = np.vstack((pred_vectors, pred_vector))

    filenames = np.array(filenames)
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    return (filenames, true_labels, pred_labels, pred_vectors)


def save_variable(var, var_name, data_dir="kdef"):
    print("save ", var_name)
    file_path = os.path.join(data_dir, var_name)
    var.tofile("{}.bin".format(file_path))


def load_variable(file_path, data_type, var_shape):
    var = np.fromfile(file_path, dtype=data_type)
    var.shape = var_shape
    return var


def calculate_acc(true_labels, pred_labels):
    true_labels.shape = pred_labels.shape
    agreement_counter = np.sum(true_labels == pred_labels)
    acc = agreement_counter / len(true_labels)

    return acc

def load_img(img_name, normalize=True, had_normalized=False):
    folder = os.path.join("kdef/original", img_name[: 4])
    img_path = os.path.join(folder, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(img_name, "is None")
    # if img_name == "BM01ANS.JPG" or img_name == "BM10ANFR.JPG":
    #     img = cv2.resize(img, (562, 762))
    img = cv2.resize(img, (32, 32))
    if normalize:
        img = np.divide(img, 255).astype("float16")

    return img


def normalize_img(img_name):
    if img_name == "AM31H.JPG":
        img_name = "AM31SUHR.JPG"
    elif img_name == "AF31V.JPG":
        img_name = "AF31SAHL.JPG"
    normalized_img = load_img(img_name, normalize=True)
    img_name_no_suffix = img_name.split(".")[0]
    if not os.path.exists("kdef/normalized_imgs"):
        os.mkdir("kdef/normalized_imgs")
    save_variable(normalized_img, img_name_no_suffix, data_dir="kdef/normalized_imgs")


def normalize_imgs(img_paths):
    outset = time.time()
    for img_path in img_paths:
        img_name = img_path.split("/")[-1]
        normalize_img(img_name)
    print("took {} min".format(round((time.time() - outset) / 60, 2)))


if __name__ == "__main__":
    api_result_dict = torch.load("kdef/output_tensor_dict.pt")
    train_img_paths = torch.load("kdef/file_list_train.pt")
    test_img_paths = torch.load("kdef/file_list_test.pt")

    normalize_imgs(train_img_paths)
    normalize_imgs(test_img_paths)

    # train_set_size = len(train_img_paths)
    # test_set_size = len(test_img_paths)
    # # train_set_size = 4076
    # # test_set_size = 815
    # (train_set_filenames, train_set_true_labels,
    #  train_set_pred_labels, train_set_pred_vectors) = load_kdef_api_results(train_img_paths, api_result_dict)
    #
    # (test_set_filenames, test_set_true_labels,
    #  test_set_pred_labels, test_set_pred_vectors) = load_kdef_api_results(test_img_paths, api_result_dict)
    #
    # train_set_acc = calculate_acc(train_set_true_labels, train_set_pred_labels)
    # test_set_acc = calculate_acc(test_set_true_labels, test_set_pred_labels)
    #
    # save_variable(train_set_filenames, "train_set_filenames")
    # save_variable(train_set_true_labels, "train_set_true_labels")
    # save_variable(train_set_pred_labels, "train_set_pred_labels")
    # save_variable(train_set_pred_vectors, "train_set_pred_vectors")
    #
    # save_variable(test_set_filenames, "test_set_filenames")
    # save_variable(test_set_true_labels, "test_set_true_labels")
    # save_variable(test_set_pred_labels, "test_set_pred_labels")
    # save_variable(test_set_pred_vectors, "test_set_pred_vectors")
    #
    # save_variable(train_set_acc, "train_set_acc")
    # save_variable(test_set_acc, "test_set_acc")
    #
    # train_set_filenames = load_variable(file_path="kdef/train_set_filenames.bin",
    #                                     data_type="<U12",
    #                                     var_shape=(train_set_size,))
    # train_set_true_labels = load_variable(file_path="kdef/train_set_true_labels.bin",
    #                                       data_type="int32",
    #                                       var_shape=(train_set_size,))
    # train_set_pred_labels = load_variable(file_path="kdef/train_set_pred_labels.bin",
    #                                       data_type="int64",
    #                                       var_shape=(train_set_size, 1))
    # train_set_pred_vectors = load_variable(file_path="kdef/train_set_pred_vectors.bin",
    #                                        data_type="float32",
    #                                        var_shape=(train_set_size, 7))
    #
    # test_set_filenames = load_variable(file_path="kdef/test_set_filenames.bin",
    #                                    data_type="<U12",
    #                                    var_shape=(test_set_size,))
    # test_set_true_labels = load_variable(file_path="kdef/test_set_true_labels.bin",
    #                                      data_type="int32",
    #                                      var_shape=(test_set_size,))
    # test_set_pred_labels = load_variable(file_path="kdef/test_set_pred_labels.bin",
    #                                      data_type="int64",
    #                                      var_shape=(test_set_size, 1))
    # test_set_pred_vectors = load_variable(file_path="kdef/test_set_pred_vectors.bin",
    #                                       data_type="float32",
    #                                       var_shape=(test_set_size, 7))
    #
    # train_set_acc = load_variable(file_path="kdef/train_set_acc.bin",
    #                               data_type="float64",
    #                               var_shape=1)[0]
    # test_set_acc = load_variable(file_path="kdef/test_set_acc.bin",
    #                              data_type="float64",
    #                              var_shape=1)[0]