import numpy as np
from PIL import Image
from skimage import measure


def load_dataset(root, dataset, split_method):
    train_txt = root + '/' + dataset + '/' + split_method + '/' + 'train.txt'
    test_txt = root + '/' + dataset + '/' + split_method + '/' + 'test.txt'
    train_img_ids = []
    val_img_ids = []
    with open(train_txt, "r") as f:
        line = f.readline()
        while line:
            train_img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    with open(test_txt, "r") as f:
        line = f.readline()
        while line:
            val_img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    return train_img_ids, val_img_ids, test_txt


def load_dataset1(root, dataset, split_method):
    test_txt = root + '/' + dataset + '/' + split_method + '/' + 'test.txt'
    val_img_ids = []
    with open(test_txt, "r") as f:
        line = f.readline()
        while line:
            val_img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    return val_img_ids, test_txt


def load_dataset_eva(root, dataset, split_method):
    train_txt = root + '/' + dataset + '/' + split_method + '/' + 'train.txt'
    test_txt = root + '/' + dataset + '/' + split_method + '/' + 'test.txt'
    #train_img_ids = []
    val_img_ids = []
    # with open(train_txt, "r") as f:
    #     line = f.readline()
    #     while line:
    #         train_img_ids.append(line.split('\n')[0])
    #         line = f.readline()
    #     f.close()
    with open(test_txt, "r") as f:
        line = f.readline()
        while line:
            val_img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    return val_img_ids, test_txt


def load_param(channel_size, backbone, blocks_per_layer=4):
    if channel_size == 'one':
        nb_filter = [4, 8, 16, 32, 64]
    elif channel_size == 'two':
        nb_filter = [8, 16, 32, 64, 128]
    elif channel_size == 'three':
        nb_filter = [16, 32, 64, 128, 256]
    elif channel_size == 'four':
        nb_filter = [32, 64, 128, 256, 512]
    elif channel_size == 'all_48':
        nb_filter = [48, 48, 48, 48, 48]
    elif channel_size == 'all_32':
        nb_filter = [32, 32, 32, 32, 32]
    elif channel_size == 'all_16':
        nb_filter = [16, 16, 16, 16, 16]

    if backbone == 'resnet_10':
        num_blocks = [1, 1, 1, 1]
    elif backbone == 'resnet_18':
        num_blocks = [2, 2, 2, 2]
    elif backbone == 'resnet_34':
        num_blocks = [3, 4, 6, 3]
    elif backbone == 'vgg_10':
        num_blocks = [1, 1, 1, 1]

    return nb_filter, num_blocks
