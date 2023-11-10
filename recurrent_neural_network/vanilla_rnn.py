import numpy as np
import tensorflow as tf

from utils.activation_functions import softmax


def rnn_cell_forward(x_t: np.ndarray, h_prev: np.ndarray, parameters):
    """
        Implements a single forward step of the RNN-cell.

    :param x_t: Input data at timestep "t", numpy array of shape (n_x, m).
    :param h_prev: Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    :param parameters: python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba  -- Bias, numpy array of shape (n_a, 1)
                        by  -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    :returns:
        a_next -- next hidden state, of shape (n_a, m)
        yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
        cache -- tuple of values needed for the backward pass, contains (a_next, a_prev, xt, parameters)
    """

    w_ax = parameters["Wax"]
    w_aa = parameters["Waa"]
    w_ya = parameters["Wya"]
    b_a = parameters["ba"]
    b_y = parameters["by"]

    h_t = np.dot(w_ax, x_t) + np.dot(w_aa, h_prev) + b_a
    a_next = np.tanh(h_t)
    o_t = np.dot(w_ya, a_next) + b_y
    yt_pred = softmax(o_t)

    cache = (a_next, h_prev.copy(), x_t, parameters)

    return a_next, yt_pred, cache


def rnn_forward(x: np.ndarray, a_0: np.ndarray, parameters):
    """
            Implement the forward propagation of the recurrent neural network described in Figure (3).

    :param x: Input data for every time-step, of shape (n_x, m, T_x).
    :param a_0: Initial hidden state, of shape (n_a, m)
    :param parameters: Python dictionary containing:
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    :returns:
        a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
        y_pred -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
        caches -- tuple of values needed for the backward pass, contains (list of caches, x)
    """
    caches = []

    n_x, m, t_x = x.shape
    n_y, n_a = parameters["Wya"].shape

    a = np.zeros(shape=(n_a, m, t_x))
    y_pred = np.zeros(shape=(n_y, m, t_x))

    a_next = a_0.copy()

    for t in range(t_x):
        a_next, yt_pred, cache = rnn_cell_forward(x_t=x[:, :, t], h_prev=a_next, parameters=parameters)
        a[:, :, t] = a_next
        y_pred[:, :, t] = yt_pred
        caches.append(cache)

    caches = (caches, x)

    return a, y_pred, caches


def check_rnn():
    np.random.seed(1)
    input_units = 3
    batch_size = 10
    time_steps = 4

    hidden_units = 5
    output_units = 2

    inputs = np.random.randn(input_units, batch_size, time_steps).astype(np.float32)
    a0 = np.random.randn(hidden_units, batch_size)

    parameters_tmp = dict()
    parameters_tmp['Waa'] = np.random.randn(hidden_units, hidden_units)
    parameters_tmp['Wax'] = np.random.randn(hidden_units, input_units)
    parameters_tmp['Wya'] = np.random.randn(output_units, hidden_units)
    parameters_tmp['ba'] = np.random.randn(hidden_units, 1)
    parameters_tmp['by'] = np.random.randn(output_units, 1)
    a_tmp, y_pred_tmp, caches_tmp = rnn_forward(inputs, a0, parameters_tmp)
    print(f"A has shape {a_tmp.shape} and mean {np.mean(a_tmp)}")
    print(f"Predicted Y has shape {y_pred_tmp.shape} and mean {np.mean(y_pred_tmp)}")

    # Tensorflow
    inputs = np.reshape(inputs, newshape=(batch_size, time_steps, input_units))
    rnn = tf.keras.layers.SimpleRNN(units=output_units, return_sequences=True)
    output = rnn(inputs)
    print(f"Tensorflow Predicted Y has shape {output.shape} and mean {np.mean(output):.2f}")


if __name__ == '__main__':
    check_rnn()
