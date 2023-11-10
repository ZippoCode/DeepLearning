import numpy as np


def normalize_rows(x: np.ndarray) -> np.ndarray:
    """
        Implement a function that normalizes each row of the matrix x (to have unit length).

    :param x: A numpy matrix of shape (n, m)
    :return: The normalized (by row) numpy matrix
    """
    x_norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    x = np.divide(x, x_norm)
    return x


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
