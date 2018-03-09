"""
mnist_loader
~~~~~~~~~~~~

A library to load weights and biases of a neural network
from a text file instead of generating them randomly
"""

#### Libraries
# Standard library
import sys
import os
import re

# Third-party libraries
import numpy as np


def load_matrices(save_file, sizes):
    """Takes the data from save_file and
    loads it into the lists weights and biases.
    """
    layer_num = len(sizes)
    if not (os.path.exists(save_file)):
        print "File %s does not exist" % (save_file)
        sys.exit(1)
    else:
        f = open(save_file, 'rU')
        raw_text = f.read()
        f.close()
        raw_text = re.split('Layer \d \w+:\n|', raw_text)
        weights = raw_text[1::2]
        biases = raw_text[2::2]

        weights = [weights[x].replace('[[', ' ').replace(']\n [', ' ').replace
                   (']]\n', ' ').replace(']]', ' ') for x in range(layer_num-1)]
        weights = [np.reshape(np.fromstring(weights[x], sep=' '), [sizes[x+1], sizes[x]]) for x in range(layer_num-1)]

        biases = [biases[x].replace('[[', ' ').replace(']\n [', ' ').replace
                  (']]\n', ' ') for x in range(layer_num-1)]
        biases = [np.reshape(np.fromstring(biases[x], sep=' '), [sizes[x+1], 1]) for x in range(layer_num-1)]
    return weights, biases


"""def main():
weights, biases = load_matrices('best_weights_and_biases_1.txt', [784, 30, 10])
print(weights)
print(biases)
print(np.shape(weights[0]))
print(np.shape(weights[1]))
print(np.shape(biases[0]))
print(np.shape(biases[1]))

if __name__ == '__main__':
main()
"""
