from utils.utils import *
from utils.public_tests import *


def test_exercise_one():
    sigmoid_derivative_test(sigmoid_derivative)
    normalizeRows_test(normalize_rows)
    softmax_test(softmax)
    L1_test(l1)
    L2_test(l2)
