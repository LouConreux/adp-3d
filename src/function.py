import math
import sys

fourth_eps = sys.float_info.epsilon**(1/4)

class SincFunction(object):
    def __init__(self, max_value, bin_size):
        self.bin_size_ = bin_size
        self.one_over_bin_size_ = 1.0 / bin_size
        self.max_value_ = max_value

    def sinc(self, x):
        # WARNING: this function might create a difference with the C++ code (precision difference)
        return sinc_pi(x, self.one_over_bin_size_, self.bin_size_)

def sinc(x, one_over_bin_size, bin_size):
    return sinc_pi(x, one_over_bin_size, bin_size)

def value2index(one_over_bin_size_, value):
    return int(value * one_over_bin_size_ + 0.5)

def sinc_pi(x, one_over_bin_size_, bin_size_):
    # index = value2index(one_over_bin_size_, x)
    # x = index * bin_size_
    if abs(x) >= 3.3 * fourth_eps:
        return math.sin(x)/x
    else:
        # |x| < (eps*120)^(1/4)
        return 1 - x * x / 6

class ExpFunction:
    def __init__(self, max_value, bin_size):
        self.bin_size_ = bin_size
        self.one_over_bin_size_ = 1.0 / bin_size
        self.max_value_ = max_value

    def exp(self, x):
        # index = value2index(self.one_over_bin_size_, abs(x))
        # y = index * self.bin_size_
        y = abs(x)
        if x < 0.0:
            return 1.0 / math.exp(y)
        return math.exp(y)
