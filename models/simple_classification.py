import copy
import typing

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegressionCV

from utils.dataloader import load_planar_dataset, load_extra_datasets
from utils.utils import sigmoid


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole grid
    z = model(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, z, cmap='Spectral')
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap='Spectral')


def train_using_logistic_regression(x: np.ndarray, y: np.ndarray, plot_result=False) -> None:
    """
        Apply a Logistic Regression on the dataset

    :param x:
    :param y:
    :param plot_result
    :return:
    """
    # Train the logistic regression classifier
    clf = LogisticRegressionCV()
    clf.fit(x.T, y.ravel())

    # Plot the decision boundary for logistic regression
    if plot_result:
        plot_decision_boundary(lambda: clf.predict(x), x, y)
        plt.title("Logistic Regression")

    # Print accuracy
    lr_predictions = clf.predict(x.T)
    accuracy_value = float(
        np.squeeze(np.dot(y, lr_predictions.T) + np.dot(1 - y, 1 - lr_predictions.T)) / float(y.size) * 100)
    print('Accuracy of logistic regression: %d ' % accuracy_value + "% (percentage of correctly labelled datapoints)")


def layer_sizes(x: np.ndarray, y: np.ndarray) -> typing.Tuple[int, int, int]:
    """

    :param x: input dataset of shape (input size, number of examples)
    :param y: labels of shape (output size, number of examples)
    :return:
        n_x -- the size of the input layer
        n_h -- the size of the hidden layer
        n_y -- the size of the output layer
    """
    n_x = x.shape[0]
    n_h = 4
    n_y = y.shape[0]
    return n_x, n_h, n_y


def initialize_parameters(n_x: int, n_h: int, n_y: int) -> dict:
    """
    :param n_x: -- size of the input layer
    :param n_h: -- size of the hidden layer
    :param n_y: -- size of the output layer

    :returns:
    params -- python dictionary containing parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    weights_one = np.random.randn(n_h, n_x) * 0.01
    bias_one = np.zeros(shape=(n_h, 1))
    weights_two = np.random.randn(n_y, n_h) * 0.01
    bias_two = np.zeros(shape=(n_y, 1))

    parameters = {"W1": weights_one,
                  "b1": bias_one,
                  "W2": weights_two,
                  "b2": bias_two}

    return parameters


def forward_propagation(x: np.ndarray, parameters: dict):
    """
        Apply Forward Propagation

    :param x: -- input data of size (n_x, m)
    :param parameters: -- python dictionary containing your parameters (output of initialization function)

    :returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """

    w1 = parameters["W1"]
    b1 = parameters["b1"]
    w2 = parameters["W2"]
    b2 = parameters["b2"]

    z1 = np.dot(w1, x) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)

    assert (a2.shape == (1, x.shape[1]))

    cache = {"Z1": z1,
             "A1": a1,
             "Z2": z2,
             "A2": a2}

    return a2, cache


def compute_cost(a2: np.ndarray, y: np.ndarray) -> float:
    """
        Computes the cross-entropy cost given in equation

    :param a2: -- The sigmoid output of the second activation, of shape (1, number of examples)
    :param y: -- "true" labels vector of shape (1, number of examples)

    :returns:
        cost -- cross-entropy cost given equation (13)
    """
    log_probs = np.multiply(y, np.log(a2)) + np.multiply((1 - y), np.log(1 - a2))
    cost = - 1 / y.shape[1] * np.sum(log_probs)
    return cost


def backward_propagation(parameters, cache, x, y) -> dict:
    """
        Implement the backward propagation using the instructions above.

    :param parameters: -- python dictionary containing our parameters
    :param cache: -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    :param x:-- input data of shape (2, number of examples)
    :param y: -- "true" labels vector of shape (1, number of examples)

    :returns:
        grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = x.shape[1]

    # First, retrieve W1 and W2 from the dictionary "parameters".
    w1 = parameters["W1"]
    w2 = parameters["W2"]

    # Retrieve also A1 and A2 from dictionary "cache".
    a1 = cache["A1"]
    a2 = cache["A2"]

    # Backward propagation
    d_z2 = a2 - y
    d_w2 = 1 / m * np.dot(d_z2, a1.T)
    db2 = 1 / m * np.sum(d_z2, axis=1, keepdims=True)
    d_z1 = np.dot(w2.T, d_z2) * (1 - np.power(a1, 2))
    d_w1 = 1 / m * np.dot(d_z1, x.T)
    db1 = 1 / m * np.sum(d_z1, axis=1, keepdims=True)

    grads = {"dW1": d_w1,
             "db1": db1,
             "dW2": d_w2,
             "db2": db2}

    return grads


def update_parameters(parameters: dict, grads: dict, learning_rate=1.2) -> dict:
    """
    Updates parameters using the gradient descent update rule given above

    :param parameters: -- python dictionary containing your parameters
    :param grads: -- python dictionary containing your gradients

    :returns:
        parameters -- python dictionary containing your updated parameters
    """
    # Retrieve a copy of each parameter from the dictionary "parameters". Use copy.deepcopy(...) for W1 and W2
    w1 = copy.deepcopy(parameters["W1"])
    b1 = copy.deepcopy(parameters["b1"])
    w2 = copy.deepcopy(parameters["W2"])
    b2 = copy.deepcopy(parameters["b2"])

    # Retrieve each gradient from the dictionary "grads"
    d_w1 = grads["dW1"]
    d_b1 = grads["db1"]
    d_w2 = grads["dW2"]
    d_b2 = grads["db2"]

    # Update rule for each parameter
    w1 = w1 - learning_rate * d_w1
    b1 = b1 - learning_rate * d_b1
    w2 = w2 - learning_rate * d_w2
    b2 = b2 - learning_rate * d_b2

    parameters = {"W1": w1,
                  "b1": b1,
                  "W2": w2,
                  "b2": b2}

    return parameters


def nn_model(x: np.ndarray, y: np.ndarray, n_h: int, num_iterations=10000, print_cost=False) -> dict:
    """
    :param x: -- dataset of shape (2, number of examples)
    :param y:-- labels of shape (1, number of examples)
    :param n_h: -- size of the hidden layer
    :param num_iterations: -- Number of iterations in gradient descent loop
    :param print_cost: -- if True, print the cost every 1000 iterations

    :returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(3)
    n_x = layer_sizes(x, y)[0]
    n_y = layer_sizes(x, y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(0, num_iterations):

        # Forward propagation
        a2, cache = forward_propagation(x, parameters)

        # Cost function
        cost = compute_cost(a2, y)

        # Backpropagation
        grads = backward_propagation(parameters, cache, x, y)

        # Gradient descent parameter update
        parameters = update_parameters(parameters, grads)

        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters


def predict(parameters: dict, x: np.ndarray, threshold=0.5) -> np.ndarray:
    """
        Using the learned parameters, predicts a class for each example in X

    :param parameters: -- python dictionary containing your parameters
    :param x: -- input data of size (n_x, m)
    :param threshold: -- threshold of prediction

    :returns:
        predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """

    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    a2, cache = forward_propagation(x, parameters)
    predictions = (a2 > threshold) * 1

    return predictions


# def show():
#     plt.figure(figsize=(16, 32))
#     hidden_layer_sizes = [1, 2, 3, 4, 5]
#
#     for i, n_h in enumerate(hidden_layer_sizes):
#         plt.subplot(5, 2, i + 1)
#         plt.title('Hidden Layer of size %d' % n_h)
#         parameters = nn_model(x, y, n_h, num_iterations=5000)
#         plot_decision_boundary(lambda lambda_: predict(parameters, lambda_.T), x, y)
#         predictions = predict(parameters, x)
#         accuracy = float(np.squeeze(np.dot(y, predictions.T) + np.dot(1 - y, 1 - predictions.T)) / float(y.size) * 100)
#         print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
#     plt.show()


def run(inputs: np.ndarray, labels: np.ndarray, visualize: bool):
    print('The shape of X is: ' + str(inputs.shape))
    print('The shape of Y is: ' + str(labels.shape))

    parameters = nn_model(x=inputs, y=labels, n_h=4, num_iterations=10000, print_cost=True)
    predictions = predict(parameters, inputs)

    accuracy = float(
        np.squeeze(np.dot(labels, predictions.T) + np.dot(1 - labels, 1 - predictions.T)) / float(labels.size) * 100)
    print('Accuracy: %d' % accuracy + '%')

    if visualize:  # Plot the decision boundary
        plt.scatter(inputs[0, :], inputs[1, :], c=labels, s=40, cmap='Spectral')
        plt.show()

        plot_decision_boundary(lambda lambda_: predict(parameters, lambda_.T), inputs, labels)
        plt.title("Decision Boundary for hidden layer size " + str(4))
        plt.show()


def main():
    x, y = load_planar_dataset(m=400)
    train_using_logistic_regression(x, y)
    run(inputs=x, labels=y, visualize=True)

    # Datasets
    noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets(n=200)

    datasets = {"noisy_circles": noisy_circles,
                "noisy_moons": noisy_moons,
                "blobs": blobs,
                "gaussian_quantiles": gaussian_quantiles}

    dataset = "gaussian_quantiles"
    X, Y = datasets[dataset]
    X, Y = X.T, Y.reshape(1, Y.shape[0])

    if dataset == "blobs":
        Y = Y % 2

    run(X, Y, True)


if __name__ == '__main__':
    main()
