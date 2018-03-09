"""
mnist_loader
~~~~~~~~~~~~

A library to save weights and biases of a neural network
in a text file for further experimentation
"""

#### Libraries
# Standard library
import sys

# Third-party libraries
import numpy as np


def save_matrices(save_file, input_weights, input_biases, num_layers, epoch):
    """Takes the data from input_weights and
    input_biases and saves it to text file
    save_file.
    """
    weights_and_biases_string = ""
    for layer in range(num_layers - 1):
        weights_and_biases_string += "Layer " + str(layer + 1) + " weights:\n" + \
                                     np.array_str(input_weights[layer], max_line_width=sys.maxint) + "\n" + \
                                     "Layer " + str(layer + 1) + " biases:\n" + \
                                     np.array_str(input_biases[layer], max_line_width=sys.maxint) + "\n"
    f = open(save_file, 'w')
    f.write(weights_and_biases_string)
    f.close()

