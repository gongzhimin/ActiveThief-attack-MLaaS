import os
import cv2
import numpy as np
from cfg import cfg


def get_true_model_predictions(img_names, dsl, labels=True):
    Y = None
    Y_idx = []
    num_classes = cfg.num_classes

    for img_name in img_names:
        Y_b = np.zeros((num_classes,), dtype="float32")
        Y_b[dsl.label_dict[img_name]] = 1
        Y_idx_b = dsl.label_dict[img_name]

        if Y is None:
            Y = np.array([Y_b])
        else:
            Y = np.vstack((Y, Y_b))

        Y_idx.append(Y_idx_b)

    if labels:
        return Y, Y_idx
    else:
        return Y



def load_variable(file_path, data_type, var_shape):
    var = np.fromfile(file_path, dtype=data_type)
    var.shape = var_shape

    return var


def load_img(img_name, normalize=True):
    if cfg.noise_dataset == "kdef":
        folder = os.path.join("kdef/original", img_name[: 4])
        img_path = os.path.join(folder, img_name)
    elif cfg.noise_dataset == "raf":
        img_path = os.path.join("raf/original", img_name)
    elif cfg.noise_dataset == 'nsfw':
        img_path = img_name
    else:
        raise Exception("No such data set type: {}".format(cfg.noise_dataset))

    img = cv2.imread(img_path)
    if normalize:
        img = np.divide(img, 255.0)
        assert (0 <= img).all() and (img <= 1.0).all()
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    resized_img = cv2.resize(crop_img, (cfg.img_size, cfg.img_size))

    return resized_img


def load_imgs(img_names):
    imgs = None
    for img_name in img_names:
        img = load_img(img_name)
        if imgs is None:
            imgs = np.array([img])
        else:
            dim = imgs.shape
            imgs = np.append(imgs, img)
            imgs = imgs.reshape(dim[0] + 1, dim[1], dim[2], dim[3])

    return imgs

