import typing

import numpy as np


def sigmoid(x: typing.Union[float, np.ndarray]) -> typing.Union[float, np.ndarray]:
    """
        Compute the sigmoid of x

    :param x: A scalar or numpy array of any size
    :return: sigmoid(x)
    """
    s = 1. / (1. + np.exp(-x))
    return s


def sigmoid_derivative(x: typing.Union[np.ndarray], z: np.ndarray) -> typing.Union[float, np.ndarray]:
    """
       Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.

    :param x: A scalar or numpy array
    :param z: 'Z' where we store for computing backward propagation efficiently


    :return: Your computed gradient.
    """

    s = 1 / (1 + np.exp(-z))
    derivative = x * s * (1 - s)

    assert x.shape == z.shape

    return derivative


def relu(x: typing.Union[float, np.ndarray]) -> typing.Union[float, np.ndarray]:
    """
        Compute the Rectified Linear Unit of x

    :param x: A scalar or numpy array of any size

    :return: ReLU(x)
    """
    s = np.maximum(0, x)
    return s


def relu_derivative(x: typing.Union[np.ndarray], z: np.ndarray) -> typing.Union[float, np.ndarray]:
    """
       Compute the gradient of the Rectified Linear Unit function with respect to its input x.

    :param x: A scalar or numpy array
    :param z: 'Z' where we store for computing backward propagation efficiently

    :return: Your computed gradient.
    """
    derivative = np.array(x, copy=True)
    derivative[z <= 0] = 0

    assert derivative.shape == z.shape

    return derivative


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
