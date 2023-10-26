import os
import sys
import typing

import h5py
import numpy as np
import sklearn
from sklearn.datasets import make_circles, make_moons, make_blobs, make_gaussian_quantiles

ROOT_DIR = sys.path[1]  # Project Root


def load_cat_dataset() -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path_train = os.path.join(ROOT_DIR, 'datasets/train_catvnoncat.h5')
    path_test = os.path.join(ROOT_DIR, 'datasets/test_catvnoncat.h5')
    if not os.path.exists(path_train) or not os.path.exists(path_test):
        raise FileNotFoundError(f"Files {path_train}, {path_train} not found")

    train_dataset = h5py.File(path_train, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # Train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # Train set labels

    test_dataset = h5py.File(path_test, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # Test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # Test set labels

    classes = np.array(test_dataset["list_classes"][:])  # The list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    assert train_set_x_orig.shape[0] == train_set_y_orig.shape[1]
    assert test_set_x_orig.shape[0] == test_set_y_orig.shape[1]
    assert train_set_x_orig.shape[1] == train_set_x_orig.shape[2]
    assert test_set_x_orig.shape[1] == test_set_x_orig.shape[2]

    num_px = train_set_x_orig.shape[1]
    print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print("Training set X shape: " + str(train_set_x_orig.shape))
    print("Training set Y shape: " + str(train_set_y_orig.shape))
    print("Testing set X shape: " + str(test_set_x_orig.shape))
    print("Testing set Y shape: " + str(test_set_y_orig.shape))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def load_planar_dataset(m: int) -> typing.Tuple[np.ndarray, np.ndarray]:
    """

    :param m: number of examples
    :return:
        - a numpy-array (matrix) X that contains your features (x1, x2)
        - a numpy-array (vector) Y that contains your labels (red:0, blue:1).
    """
    np.random.seed(1)
    n = int(m / 2)  # number of points per class
    d = 2  # dimensionality
    x = np.zeros((m, d))  # data matrix where each row is a single example
    y = np.zeros((m, 1), dtype='uint8')  # labels vector (0 for red, 1 for blue)
    a = 4  # maximum ray of the flower

    for j in range(2):
        ix = range(n * j, n * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, n) + np.random.randn(n) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(n) * 0.2  # radius
        x[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    x = x.T
    y = y.T
    assert x.shape[1] == y.shape[1]

    return x, y


def load_extra_datasets(n: int):
    noisy_circles = make_circles(n_samples=n, factor=.5, noise=.3)
    noisy_moons = make_moons(n_samples=n, noise=.2)
    blobs = make_blobs(n_samples=n, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = make_gaussian_quantiles(mean=None, cov=0.5, n_samples=n, n_features=2,
                                                 n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(n, 2), np.random.rand(n, 2)

    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure
