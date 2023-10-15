import os.path
import typing

import h5py
import sys
import numpy as np

ROOT_DIR = sys.path[1]  # Project Root


def load_dataset():
    path_train = os.path.join(ROOT_DIR, 'datasets/train_catvnoncat.h5')
    path_test = os.path.join(ROOT_DIR, 'datasets/test_catvnoncat.h5')
    if not os.path.exists(path_train) or not os.path.exists(path_test):
        return

    train_dataset = h5py.File(path_train, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File(path_test, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def sigmoid(x: typing.Union[float, np.ndarray]) -> typing.Union[float, np.ndarray]:
    """
        Compute the sigmoid of x

    :param x: A scalar or numpy array of any size
    :return: sigmoid(x)
    """
    s = 1 / (1 + np.exp(-x))
    return s


def sigmoid_derivative(x: typing.Union[float, np.ndarray]) -> typing.Union[float, np.ndarray]:
    """
       Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.

    :param x: A scalar or numpy array
    :return: Your computed gradient.
    """
    s = 1 / (1 + np.exp(-x))
    ds = s * (1 - s)
    return ds


def normalize_rows(x: np.ndarray) -> np.ndarray:
    """
        Implement a function that normalizes each row of the matrix x (to have unit length).

    :param x: A numpy matrix of shape (n, m)
    :return: The normalized (by row) numpy matrix
    """
    x_norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    x = np.divide(x, x_norm)
    return x


def softmax(x: np.ndarray) -> np.ndarray:
    """
        Calculates the softmax for each row of the input x.

    :param x: A numpy matrix of shape (n, m)
    :return: A numpy matrix equal to the softmax of x, of shape (m,n)
    """
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = np.divide(x_exp, x_sum)
    return s


def l1(yhat: np.ndarray, y: np.ndarray) -> float:
    """

    :param yhat: vector of size m (predicted labels)
    :param y: vector of size m (true labels)
    :return: the value of the L1 loss function defined above
    """

    loss = np.sum(np.abs(yhat - y))

    return loss


def l2(yhat: np.ndarray, y: np.ndarray) -> float:
    """

    :param yhat: vector of size m (predicted labels)
    :param y:  vector of size m (true labels)
    :return: the value of the L2 loss function defined above
    """
    loss = np.sum(np.dot(y - yhat, y - yhat))
    return loss