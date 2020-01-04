import argparse
import sys
import time
import timeit

import numpy as np
import pandas as pd

from sklearn import datasets, metrics, svm


def load_mnist_data():
    # The digits dataset
    digits = datasets.load_digits()

    # To apply a classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    return data, digits.target


def load_cifar10_data():
    data = list()
    labels = list()
    for batch_index in range(1, 6):
        batch = unpickle("cifar-10-batches-py/data_batch_" + str(batch_index))
        data = data + list(batch[b"data"])
        labels = labels + batch[b"labels"]
    return np.array(data), np.array(labels)


def load_cifar100_data():
    data = list()
    labels = list()
    for batch_name in ["test", "train"]:
        batch = unpickle("cifar-100/" + batch_name)
        data = data + list(batch[b"data"])
        labels = labels + batch[b"coarse_labels"]
    return np.array(data), np.array(labels)


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_letter_data():
    data = pd.read_csv(
        "letter-recognition/letter-recognition.data", header=None)
    y = np.array([ord(x) - 65 for x in data.iloc[:, 0]])
    X = np.array(data.iloc[:, 1:len(data)])
    return X, y
