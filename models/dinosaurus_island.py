import copy
import typing

import numpy as np

from utils.tmp_utils_to_delete import rnn_forward, rnn_backward, update_parameters, initialize_parameters, \
    get_initial_loss, smooth, get_sample
from utils.utils import softmax


def build_vocabulary() -> typing.Tuple[str, typing.Dict, typing.Dict]:
    """

    :return:
        char_to_ix: A Python dictionary to map each character to an index from 0-26.
        ix_to_char: A Python dictionary that maps each index back to the corresponding character.
    """
    text = open('../datasets/texts/dinosaurus.txt', 'r').read()
    text = text.lower()
    chars = list(set(text))
    chars = sorted(chars)
    data_size, vocab_size = len(text), len(chars)
    print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))

    # Build dictionary
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    return text, char_to_idx, ix_to_char


def clip(gradients: dict, max_value: float):
    """
        Clips the values of gradients between minimum and maximum.

    :parameter gradients: A dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
    :parameter max_value: Everything above this number is set to this number, and everything less than
                            maxValue is set to maxValue

    :return:
        gradients -- A dictionary with the clipped gradients.
    """
    gradients = copy.deepcopy(gradients)

    d_waa = gradients['dWaa']
    d_wax = gradients['dWax']
    d_wya = gradients['dWya']
    db = gradients['db']
    dby = gradients['dby']

    for gradient in [d_wax, d_waa, d_wya, db, dby]:
        np.clip(gradient, -max_value, max_value, out=gradient)

    gradients = {"dWaa": d_waa, "dWax": d_wax, "dWya": d_wya, "db": db, "dby": dby}

    return gradients


# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: sample

def sample(parameters: typing.Dict, char_to_ix: typing.Dict, seed: int):
    """
        Sample a sequence of characters according to a sequence of probability distributions output of the RNN

    :parameter parameters: Python dictionary containing the parameters Waa, Wax, Wya, by, and b.
    :parameter char_to_ix: Python dictionary mapping each character to an index.
    :parameter seed: Used for grading purposes. Do not worry about it.

    :returns:
        indices -- A list of length n containing the indices of the sampled characters.
    """

    w_ax = parameters["Wax"]
    w_aa = parameters["Waa"]
    w_ya = parameters["Wya"]
    by = parameters["by"]
    b = parameters["b"]

    vocab_size = by.shape[0]
    n_a = w_aa.shape[1]

    x = np.zeros(shape=(vocab_size, 1))
    a_prev = np.zeros(shape=(n_a, 1))
    indices = []
    idx = -1

    counter = 0
    newline_character = char_to_ix['\n']

    while idx != newline_character and counter != 50:
        a = np.tanh(np.dot(w_ax, x) + np.dot(w_aa, a_prev) + b)
        z = np.dot(w_ya, a) + by
        y = softmax(z)

        # Sampling
        np.random.seed(counter + seed)
        idx = np.random.choice(range(len(y)), p=y.ravel())

        # Update
        indices.append(idx)
        x = np.zeros(shape=(vocab_size, 1))
        x[idx] = 1

        a_prev = a

        seed += 1
        counter += 1

    if counter == 50:
        indices.append(char_to_ix['\n'])

    return indices


# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: optimize

def optimize(X, Y, a_prev, parameters, learning_rate=0.01):
    """
    Execute one step of the optimization to train the model.

    Arguments:
    X -- list of integers, where each integer is a number that maps to a character in the vocabulary.
    Y -- list of integers, exactly the same as X but shifted one index to the left.
    a_prev -- previous hidden state.
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        b --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    learning_rate -- learning rate for the model.

    Returns:
    loss -- value of the loss function (cross-entropy)
    gradients -- python dictionary containing:
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
                        db -- Gradients of bias vector, of shape (n_a, 1)
                        dby -- Gradients of output bias vector, of shape (n_y, 1)
    a[len(X)-1] -- the last hidden state, of shape (n_a, 1)
    """

    loss, cache = rnn_forward(X, Y, a_prev, parameters)
    gradients, a = rnn_backward(X, Y, parameters, cache)
    gradients = clip(gradients, max_value=5)
    parameters = update_parameters(parameters, gradients, learning_rate)

    return loss, gradients, a[len(X) - 1]


# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: model

def model(data_x, ix_to_char, char_to_ix, num_iterations=35000, n_a=50, dino_names=7, vocab_size=27, verbose=False):
    """
    Trains the model and generates dinosaur names.

    Arguments:
    data_x -- text corpus, divided in words
    ix_to_char -- dictionary that maps the index to a character
    char_to_ix -- dictionary that maps a character to an index
    num_iterations -- number of iterations to train the model for
    n_a -- number of units of the RNN cell
    dino_names -- number of dinosaur names you want to sample at each iteration.
    vocab_size -- number of unique characters found in the text (size of the vocabulary)

    Returns:
    parameters -- learned parameters
    """

    # Retrieve n_x and n_y from vocab_size
    n_x, n_y = vocab_size, vocab_size

    # Initialize parameters
    parameters = initialize_parameters(n_a, n_x, n_y)

    # Initialize loss (this is required because we want to smooth our loss)
    loss = get_initial_loss(vocab_size, dino_names)

    # Build list of all dinosaur names (training examples).
    examples = [x.strip() for x in data_x]

    # Shuffle list of all dinosaur names
    np.random.seed(0)
    np.random.shuffle(examples)

    # Initialize the hidden state of your LSTM
    a_prev = np.zeros((n_a, 1))

    # for grading purposes
    last_dino_name = "abc"

    # Optimization loop
    for j in range(num_iterations):

        # Set the index `idx` (see instructions above)
        idx = j % len(examples)

        # Set the input X (see instructions above)
        single_example = idx
        single_example_chars = [c for c in examples[single_example]]
        single_example_ix = [char_to_ix[c] for c in single_example_chars]
        X = [None] + single_example_ix

        # Set the labels Y (see instructions above)
        ix_newline = [char_to_ix["\n"]]
        Y = X[1:] + ix_newline

        # Perform one optimization step: Forward-prop -> Backward-prop -> Clip -> Update parameters
        # Choose a learning rate of 0.01
        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters, learning_rate=0.01)

        # debug statements to aid in correctly forming X, Y
        if verbose and j in [0, len(examples) - 1, len(examples)]:
            print("j = ", j, "idx = ", idx, )
        if verbose and j in [0]:
            print("single_example =", single_example)
            print("single_example_chars", single_example_chars)
            print("single_example_ix", single_example_ix)
            print(" X = ", X, "\n", "Y =       ", Y, "\n")

        # to keep the loss smooth.
        loss = smooth(loss, curr_loss)

        # Every 2000 Iteration, generate "n" characters thanks to sample() to check if the model is learning properly
        if j % 2000 == 0:

            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')

            # The number of dinosaur names to print
            seed = 0
            for name in range(dino_names):
                # Sample indices and print them
                sampled_indices = sample(parameters, char_to_ix, seed)
                last_dino_name = get_sample(sampled_indices, ix_to_char)
                print(last_dino_name.replace('\n', ''))

                seed += 1  # To get the same result (for grading purposes), increment the seed by one.

            print('\n')

    return parameters, last_dino_name


def main():
    text, char_to_idx, ix_to_char = build_vocabulary()
    parameters, last_name = model(text.split("\n"), ix_to_char, char_to_idx, 22001, verbose=True)


if __name__ == '__main__':
    main()
