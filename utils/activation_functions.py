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

    :return: The computed gradient.
    """
    derivative = np.array(x, copy=True)
    derivative[z <= 0] = 0

    assert derivative.shape == z.shape

    return derivative


def softmax(x: np.ndarray) -> np.ndarray:
    """
        Calculates the softmax for each row of the input x.

    :param x: A numpy matrix of shape (n, m)
    :return: A numpy matrix equal to the softmax of x, of shape (m,n)
    """
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=0, keepdims=True)
    s = np.divide(x_exp, x_sum)
    return s
