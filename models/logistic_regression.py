import copy
import sys
import os
import typing
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils.dataloader import load_cat_dataset
from utils.utils import sigmoid

ROOT_DIR = sys.path[1]  # Project Root


def flatten_and_standardize(train_set_x_orig: np.ndarray, test_set_x_orig: np.ndarray) -> (
        typing.Tuple)[np.ndarray, np.ndarray]:
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    train_set = train_set_x_flatten / 255.
    test_set = test_set_x_flatten / 255.
    return train_set, test_set


def initialize_with_zeros(dim: int) -> typing.Tuple[np.ndarray, float]:
    """
        This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    :param dim: size of the w vector we want (or number of parameters in this case)
    :returns:
        w  initialized vector of shape (dim, 1)
        b  initialized scalar (corresponds to the bias) of type float
    """
    w = np.zeros(shape=(dim, 1))
    b = 0.
    return w, b


def propagate(weights: np.ndarray, bias: float, x: np.ndarray, y: np.ndarray) -> typing.Tuple[dict, float]:
    """
        Implement the cost function and its gradient for the propagation explained above.

    :param weights: weights, a numpy array of size (num_px * num_px * 3, 1)
    :param bias: bias, a scalar
    :param x: data of size (num_px * num_px * 3, number of examples)
    :param y: true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    :returns:
        grads dictionary containing the gradients of the weights and bias
                (dw  gradient of the loss with respect to w, thus same shape as w)
                (db  gradient of the loss with respect to b, thus same shape as b)
        cost  negative log-likelihood cost for logistic regression
    """

    m = x.shape[1]
    a = sigmoid(np.dot(weights.T, x) + bias)
    cost = - 1 / m * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))

    dw = 1 / m * np.dot(x, (a - y).T)
    db = 1 / m * np.sum(a - y)
    # cost = np.squeeze(np.array(cost))

    grads = {"dw": dw,
             "db": db}

    return grads, cost


def optimize(weights: np.ndarray, biases: float, x: np.ndarray, y: np.ndarray, num_iterations=100, learning_rate=0.009,
             print_cost=False) -> typing.Tuple[dict, dict, list]:
    """
        This function optimizes w and b by running a gradient descent algorithm

    :param weights: weights, a numpy array of size (num_px * num_px * 3, 1)
    :param biases: bias, a scalar
    :param x: data of shape (num_px * num_px * 3, number of examples)
    :param y: true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    :param num_iterations: number of iterations of the optimization loop
    :param learning_rate: learning rate of the gradient descent update rule
    :param print_cost: True to print the loss every 100 steps

    :returns:
        params dictionary containing the weights w and bias b
        grads dictionary containing the gradients of the weights and bias with respect to the cost function
        costs list of all the costs computed during the optimization, this will be used to plot the learning curve.
    """

    weights = copy.deepcopy(weights)
    biases = copy.deepcopy(biases)

    costs = []

    dw, db = 0., 0.
    for i in range(num_iterations):
        grads, cost = propagate(weights, biases, x, y)
        dw = grads["dw"]
        db = grads["db"]

        weights = weights - learning_rate * dw
        biases = biases - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print("Loss after iteration %i: %f" % (i, cost))

    params = {"w": weights, "b": biases}
    grads = {"dw": dw, "db": db}

    return params, grads, costs


def predict(weights: np.ndarray, bias: float, x: np.ndarray):
    """
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    :param weights: weights, a numpy array of size (num_px * num_px * 3, 1)
    :param bias: bias, a scalar
    :param x: data of size (num_px * num_px * 3, number of examples)
    :return:
        y_prediction  a numpy array (vector) containing all predictions (0/1) for the examples in X
    """

    m = x.shape[1]
    y_prediction = np.zeros((1, m))
    weights = weights.reshape(x.shape[0], 1)
    a = sigmoid(np.dot(weights.T, x) + bias)
    for i in range(a.shape[1]):
        if a[0, i] > 0.5:
            y_prediction[0, i] = 1
        else:
            y_prediction[0, i] = 0
    return y_prediction


def model(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, num_iterations=2000,
          learning_rate=0.5, print_cost=False) -> dict:
    """
        Builds the Logistic Regression model by calling the function you've implemented previously

    :param x_train: -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    :param y_train: -- training labels represented by a numpy array (vector) of shape (1, m_train)
    :param x_test: -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    :param y_test: -- test labels represented by a numpy array (vector) of shape (1, m_test)
    :param num_iterations: -- hyperparameter representing the number of iterations to optimize the parameters
    :param learning_rate: -- hyperparameter representing the learning rate used in the update rule of optimize()
    :param print_cost: -- Set to True to print the cost every 100 iterations

    Returns:
        d -- dictionary containing information about the model.
    """
    w, b = initialize_with_zeros(x_train.shape[0])
    params, grads, costs = optimize(w, b, x_train, y_train, num_iterations, learning_rate, print_cost)
    w = params["w"]
    b = params["b"]

    y_prediction_test = predict(w, b, x_test)
    y_prediction_train = predict(w, b, x_train)

    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    logistic_regression_model = {
        "costs": costs,
        "Y_prediction_test": y_prediction_test,
        "Y_prediction_train": y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations}

    return logistic_regression_model


def show_different_learning_rate(learning_rates: list, train_set_x: np.ndarray, train_set_y: np.ndarray,
                                 test_set_x: np.ndarray, test_set_y: np.ndarray):
    models = {}
    for lr in learning_rates:
        print("Training a model with learning rate: " + str(lr))
        models[str(lr)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500, learning_rate=lr,
                                print_cost=False)
        plt.plot(np.squeeze(models[str(lr)]["costs"]), label=str(models[str(lr)]["learning_rate"]))

    plt.ylabel('cost')
    plt.xlabel('iterations (hundreds)')
    legend = plt.legend(loc='upper right', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()


def predict_image(logistic_regression_model: dict, image_path: Path, patch_size: int, classes: np.ndarray):
    if not image_path.exists():
        print(f"Image \"{image_path}\" not exist")
        return

    image = np.array(Image.open(image_path).resize((patch_size, patch_size)))
    plt.imshow(image)
    plt.show()
    image = image / 255.
    image = image.reshape((1, patch_size * patch_size * 3)).T
    my_predicted_image = predict(logistic_regression_model["w"], logistic_regression_model["b"], image)

    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[
        int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")


def main():
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_cat_dataset()
    train_set_x, test_set_x = flatten_and_standardize(train_set_x_orig, test_set_x_orig)
    print("Training set X shape: " + str(train_set_x.shape))
    print("Training set Y shape: " + str(train_set_y.shape))
    print("Testing set X shape: " + str(test_set_x.shape))
    print("Testing set Y shape: " + str(test_set_y.shape))

    print("Train model")
    logistic_regression_model = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000,
                                      learning_rate=0.005, print_cost=True)
    costs = np.squeeze(logistic_regression_model['costs'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(logistic_regression_model["learning_rate"]))
    plt.show()

    learning_rates = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    show_different_learning_rate(learning_rates, train_set_x, train_set_y, test_set_x, test_set_y)
    image_size = train_set_x_orig.shape[1]
    image_path = os.path.join(ROOT_DIR, 'datasets/images/cat_1.jpg')
    predict_image(logistic_regression_model, Path(image_path), patch_size=image_size, classes=classes)


if __name__ == '__main__':
    main()
