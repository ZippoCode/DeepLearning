import copy
import sys

import matplotlib.pyplot as plt
import numpy as np

from models.logistic_regression import flatten_and_standardize
from utils.dataloader import load_cat_dataset
from utils.initializers import initialize_parameters_neural_network
from utils.utils import sigmoid, relu, relu_derivative, sigmoid_derivative


def linear_activation_forward(inputs: np.ndarray, weights: np.ndarray, bias: np.ndarray, activation: str):
    """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

    :param inputs: Activations from previous layer (or input data): (size of previous layer, number of examples)
    :param weights: Weights matrix: numpy array of shape (size of current layer, size of previous layer)
    :param bias: Bias vector, numpy array of shape (size of the current layer, 1)
    :param activation: The activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    :returns:
        A -- the output of the activation function, also called the post-activation value
        cache -- a python tuple containing "linear_cache" and "activation_cache";
                 stored for computing the backward pass efficiently
    """
    activations = None

    z = np.dot(weights, inputs) + bias
    linear_cache = (inputs, weights, bias)

    if activation == "sigmoid":
        activations = sigmoid(z)

    elif activation == "relu":
        activations = relu(z)

    activation_cache = copy.deepcopy(z)
    cache = (linear_cache, activation_cache)

    return activations, cache


def model_forward(inputs: np.ndarray, parameters: dict):
    """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    :param inputs: Data, numpy array of shape (input size, number of examples)
    :param parameters: Output of initialize_parameters_deep()

    :returns:
        y_hat  -- activation value from the output (last) layer
        caches -- list of caches containing:
                    every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
    """
    caches = []
    activations = inputs
    num_layers = len(parameters) // 2

    for layer in range(1, num_layers):
        previous_activations = activations
        weights = parameters["W" + str(layer)]
        bias = parameters["b" + str(layer)]
        activations, cache = linear_activation_forward(previous_activations, weights, bias, 'relu')
        caches.append(cache)

    weights = parameters["W" + str(num_layers)]
    bias = parameters["b" + str(num_layers)]
    y_hat, cache = linear_activation_forward(activations, weights, bias, 'sigmoid')
    caches.append(cache)

    return y_hat, caches


def compute_cost(probabilities: np.ndarray, targets: np.ndarray):
    """
    Implement the cost function defined by equation (7).

    :param probabilities: -- probability vector corresponding to label predictions
    :param targets: -- true "label" vector

    :returns:
        cost -- cross-entropy cost
    """
    assert probabilities.shape == targets.shape

    m = targets.shape[1]
    cost = - 1. / m * np.sum(targets * np.log(probabilities) + (1 - targets) * np.log(1 - probabilities))
    cost = np.squeeze(cost)

    return cost


def linear_activation_backward(d_a, cache, activation):
    """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.

    :param d_a: Post-activation gradient for current layer l
    :param cache: Tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    :param activation: The activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    :returns:
        d_a_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        d_w -- Gradient of the cost with respect to W (current layer l), same shape as W
        d_b -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    d_z = np.ndarray(shape=d_a.shape)
    linear_cache, activation_cache = cache

    a_prev, weights, bias = linear_cache
    m = a_prev.shape[1]

    if activation == "relu":
        d_z = relu_derivative(d_a, activation_cache)

    elif activation == "sigmoid":
        d_z = sigmoid_derivative(d_a, activation_cache)

    d_a_prev = np.dot(weights.T, d_z)
    d_w = 1. / m * np.dot(d_z, a_prev.T)
    d_b = 1. / m * np.sum(d_z, axis=1, keepdims=True)

    return d_a_prev, d_w, d_b


def model_backward(y_hat, y, caches):
    """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    :param y_hat: Probability vector, output of the forward propagation (L_model_forward())
    :param y: True "label" vector (containing 0 if non-cat, 1 if cat)
    :param caches: List of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1))
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    :returns:
        grads -- A dictionary with the gradients { grads["dA" + str(l)], grads["dW" + str(l)] , grads["db" + str(l)]}
    """
    assert y_hat.shape == y.shape
    grads = {}
    num_layers = len(caches)
    y = y.reshape(y_hat.shape)

    # Initializing the backpropagation
    derivative_y_hat = - (np.divide(y, y_hat) - np.divide(1 - y, 1 - y_hat))

    current_cache = caches[num_layers - 1]
    d_a_prev_temp, d_w_temp, db_temp = linear_activation_backward(derivative_y_hat, current_cache, activation="sigmoid")
    grads["dA" + str(num_layers - 1)] = d_a_prev_temp
    grads["dW" + str(num_layers)] = d_w_temp
    grads["db" + str(num_layers)] = db_temp

    for layer in reversed(range(num_layers - 1)):
        current_cache = caches[layer]
        d_a_prev_temp, d_w_temp, db_temp = linear_activation_backward(d_a_prev_temp, current_cache, activation="relu")
        grads["dA" + str(layer)] = d_a_prev_temp
        grads["dW" + str(layer + 1)] = d_w_temp
        grads["db" + str(layer + 1)] = db_temp

    return grads


def update_parameters(parameters: dict, grads: dict, learning_rate: float):
    """
        Update parameters using gradient descent


    :param parameters: Dictionary containing your parameters
    :param grads: Dictionary containing your gradients, output of L_model_backward
    :param learning_rate: The learning rate value

    :returns:
        parameters Python dictionary containing updated parameters
    """
    parameters = copy.deepcopy(parameters)
    num_layers = len(parameters) // 2

    for l in range(num_layers):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


def layer_model(inputs: np.ndarray, targets: np.ndarray, layers_dims: list, learning_rate=0.0075, num_iterations=3000,
                print_cost=False):
    """
        Implements a layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    :param inputs: Input data, of shape (n_x, number of examples)
    :param targets: True "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
    :param layers_dims: List containing the input size and each layer size, of length (number of layers + 1).
    :param learning_rate: Learning rate of the gradient descent update rule
    :param num_iterations: Number of iterations of the optimization loop
    :param print_cost: If True, it prints the cost every 100 steps

    :returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []  # keep track of cost

    # Parameters initialization.
    parameters = initialize_parameters_neural_network(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        predictions, caches = model_forward(inputs, parameters)

        # Compute cost.
        cost = compute_cost(predictions, targets)

        # Backward propagation.
        grads = model_backward(y_hat=predictions, y=targets, caches=caches)

        # Update parameters.
        parameters = update_parameters(parameters=parameters, grads=grads, learning_rate=learning_rate)

        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)

    return parameters, costs


def plot_costs(costs, learning_rate=0.0075):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


def predict(inputs: np.ndarray, targets: np.ndarray, parameters: dict):
    """
        This function is used to predict the results of a layer neural network.

    :param inputs: Data set of examples
    :param targets: Target data set
    :param parameters: -- parameters of the trained model

    :returns:
        p -- predictions for the given dataset X
    """

    m = inputs.shape[1]
    p = np.zeros((1, m))

    # Forward propagation
    probabilities, caches = model_forward(inputs, parameters)

    for i in range(0, probabilities.shape[1]):
        if probabilities[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    print("Accuracy: " + str(np.sum((p == targets) / m)))

    return p


def main():
    train_x, train_y, test_x, test_y, classes = load_cat_dataset()
    train_x, test_x = flatten_and_standardize(train_x, test_x)
    print("Train X shape after flatten: " + str(train_x.shape))
    print("Test X shape after flatten: " + str(test_x.shape))

    layers_dims = [12288, 21, 7, 5, 1]
    parameters, costs = layer_model(train_x, train_y, layers_dims, num_iterations=2500, print_cost=True)

    plot_costs(costs, learning_rate=0.0075)

    predict(train_x, train_y, parameters)
    predict(test_x, test_y, parameters)


if __name__ == '__main__':
    main()
    sys.exit()
