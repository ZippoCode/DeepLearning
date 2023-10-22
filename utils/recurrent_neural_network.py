import copy
import typing

import numpy as np

from utils.utils import softmax, sigmoid


def rnn_cell_forward(xt: np.ndarray, a_prev: np.ndarray, parameters):
    """
        Implements a single forward step of the RNN-cell.
        
    :param xt: Input data at timestep "t", numpy array of shape (n_x, m).
    :param a_prev: Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    :param parameters: python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    :returns:
        a_next -- next hidden state, of shape (n_a, m)
        yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
        cache -- tuple of values needed for the backward pass, contains (a_next, a_prev, xt, parameters)
    """

    w_ax = parameters["Wax"]
    w_aa = parameters["Waa"]
    w_ya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    a_next = np.tanh(np.dot(w_ax, xt) + np.dot(w_aa, a_prev) + ba)
    yt_pred = softmax(np.dot(w_ya, a_next) + by)

    cache = (a_next, a_prev, xt, parameters)

    return a_next, yt_pred, cache


def rnn_forward(x: np.ndarray, a0: np.ndarray, parameters):
    """
            Implement the forward propagation of the recurrent neural network described in Figure (3).

    :param x: Input data for every time-step, of shape (n_x, m, T_x).
    :param a0: Initial hidden state, of shape (n_a, m)
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
    # Initialize "caches" which will contain the list of all caches
    caches = []

    # Retrieve dimensions from shapes of x and parameters["Wya"]
    n_x, m, t_x = x.shape
    n_y, n_a = parameters["Wya"].shape

    # initialize "a" and "y_pred" with zeros (≈2 lines)
    a = np.zeros(shape=(n_a, m, t_x))
    y_pred = np.zeros(shape=(n_y, m, t_x))

    # Initialize a_next (≈1 line)
    a_next = copy.deepcopy(a0)

    # loop over all time-steps
    for t in range(t_x):
        # Update next hidden state, compute the prediction, get the cache (≈1 line)
        a_next, yt_pred, cache = rnn_cell_forward(xt=x[:, :, t], a_prev=a_next, parameters=parameters)
        # Save the value of the new "next" hidden state in a (≈1 line)
        a[:, :, t] = a_next
        # Save the value of the prediction in y (≈1 line)
        y_pred[:, :, t] = yt_pred
        # Append "cache" to "caches" (≈1 line)
        caches.append(cache)

    # store values needed for backward propagation in cache
    caches = (caches, x)

    return a, y_pred, caches


def lstm_cell_forward(xt: np.ndarray, a_prev: np.ndarray, c_prev: np.ndarray, parameters: typing.Dict):
    """
        Implement a single forward step of the LSTM-cell.

    :param xt: Input data at timestep "t", numpy array of shape (n_x, m).
    :param a_prev: Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    :param c_prev: Memory state at timestep "t-1", numpy array of shape (n_a, m)
    :param parameters: Python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo --  Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    :returns:
            a_next -- next hidden state, of shape (n_a, m)
            c_next -- next memory state, of shape (n_a, m)
            yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
            cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt,
                        parameters)
    """

    # Retrieve parameters from "parameters"
    Wf = parameters["Wf"]  # forget gate weight
    bf = parameters["bf"]
    Wi = parameters["Wi"]  # update gate weight (notice the variable name)
    bi = parameters["bi"]  # (notice the variable name)
    Wc = parameters["Wc"]  # candidate value weight
    bc = parameters["bc"]
    Wo = parameters["Wo"]  # output gate weight
    bo = parameters["bo"]
    Wy = parameters["Wy"]  # prediction weight
    by = parameters["by"]

    # Retrieve dimensions from shapes of xt and Wy
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    # Concatenate a_prev and xt (≈1 line)
    concat = np.concatenate([a_prev, xt], axis=0)

    # Compute values for ft, it, cct, c_next, ot, a_next using the formulas given figure (4) (≈6 lines)
    ft = sigmoid(np.dot(Wf, concat) + bf)
    it = sigmoid(np.dot(Wi, concat) + bi)
    cct = np.tanh(np.dot(Wc, concat) + bc)
    c_next = np.multiply(ft, c_prev) + np.multiply(it, cct)
    ot = sigmoid(np.dot(Wo, concat) + bo)
    a_next = np.multiply(ot, np.tanh(c_next))

    # Compute prediction of the LSTM cell (≈1 line)
    yt_pred = softmax(np.dot(Wy, a_next) + by)

    # store values needed for backward propagation in cache
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache


# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: lstm_forward

def lstm_forward(x: np.ndarray, a0: np.ndarray, parameters: typing.Dict):
    """
        Implement the forward propagation of the recurrent neural network using an LSTM-cell.

    :param x: Input data for every time-step, of shape (n_x, m, T_x).
    :param a0: Initial hidden state, of shape (n_a, m)
    :param parameters: python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc -- Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo -- Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    :returns:
        a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
        y -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
        c -- The value of the cell state, numpy array of shape (n_a, m, T_x)
        caches -- tuple of values needed for the backward pass, contains (list of all the caches, x)
    """

    # Initialize "caches", which will track the list of all the caches
    caches = []

    Wy = parameters['Wy']
    # Retrieve dimensions from shapes of x and parameters['Wy'] (≈2 lines)
    n_x, m, T_x = x.shape
    n_y, n_a = parameters['Wy'].shape

    # initialize "a", "c" and "y" with zeros (≈3 lines)
    a = np.zeros(shape=(n_a, m, T_x))
    c = np.zeros(shape=(n_a, m, T_x))
    y = np.zeros(shape=(n_y, m, T_x))

    # Initialize a_next and c_next (≈2 lines)
    a_next = copy.deepcopy(a0)
    c_next = copy.deepcopy(c[:, :, 0])

    # loop over all time-steps
    for t in range(T_x):
        # Get the 2D slice 'xt' from the 3D input 'x' at time step 't'
        xt = x[:, :, t]
        # Update next hidden state, next memory state, compute the prediction, get the cache (≈1 line)
        a_next, c_next, yt, cache = lstm_cell_forward(xt=xt, a_prev=a_next, c_prev=c_next, parameters=parameters)
        # Save the value of the new "next" hidden state in a (≈1 line)
        a[:, :, t] = a_next
        # Save the value of the next cell state (≈1 line)
        c[:, :, t] = c_next
        # Save the value of the prediction in y (≈1 line)
        y[:, :, t] = yt
        # Append the cache into caches (≈1 line)
        caches.append(cache)

    # store values needed for backward propagation in cache
    caches = (caches, x)

    return a, y, c, caches
