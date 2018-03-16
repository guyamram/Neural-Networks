#!/usr/bin/python


#### Libraries
# Standard library
import sys
import time
import random

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt

# Helper libraries
import mnist_loader
import save_matrices
import load_matrices
import overfitting as ofit

np.set_printoptions(threshold=sys.maxint)


class QuadraticCost(object):
    @staticmethod
    def cost(emp_res, exp_res):
        # returns the quadratic cost of the output of the neural network
        return 0.5*np.linalg.norm(emp_res-exp_res)**2

    @staticmethod
    def delta(emp_res, exp_res, weighted_inputs):
        # returns the delta of the output layer
        return (emp_res-exp_res)*derived_sigmoid_function(weighted_inputs)


class CrossEntropyCost(object):
    @staticmethod
    def cost(emp_res, exp_res):
        # calculates the cross entropy cost function of the empirical result vs the expected results
        length = len(exp_res)
        if length != len(emp_res):
            print "Error calculating cost function - different length results"
            sys.exit(1)
        else:
            return np.nan_to_num(-np.dot(exp_res, np.log(emp_res)) - np.dot(1 - exp_res, np.log(1 - emp_res)))

    @staticmethod
    def delta(emp_res, exp_res, weighted_inputs):
        # returns the delta of the output layer
        return emp_res-exp_res


class NeuralNetwork(object):
    # a collection of perceptrons, designed to learn through gradient descent.
    def __init__(self, sizes, load_file=None, cost=CrossEntropyCost):  # DONE add a feature for starting with given weight matrices
        self.sizes      = sizes
        self.num_layers = len(sizes)
        self.cost = cost
        if load_file:
            self.weights, self.biases = load_matrices.load_matrices(load_file, self.sizes)
        else:
            self.biases     = [np.random.randn(x, 1) for x in self.sizes[1:]]
            self.weights    = [np.random.randn(x, y) #/ np.sqrt(y)
                               for (x, y) in zip(self.sizes[1:], self.sizes[:-1])]
        # add a feature for making sure we don't get too much symmetry
        # i.e a matrix of all one number or two identical rows etc.

    def gradient_descent(self, training_input, epochs,
                         mini_batch_size, learning_factor,
                         lmbda=0.0,
                         test_input=None,
                         save_file=None,
                         monitor_test_accuracy=False,
                         monitor_test_cost=False,
                         monitor_training_accuracy=False,
                         monitor_training_cost=False):
        # this function trains the network through stochastic gradient
        # descent, calling on back_prop to calculate the new weights.
        # [training_input, validation_input, test_input] = mnist_loader.load_data()
        best_results = [0, 10000, 0, 10000]
        best_epochs = [0, 0, 0, 0]
        average_test_cost = []
        average_test_accuracy = []
        average_training_cost = []
        average_training_accuracy = []
        for epoch in range(epochs):
            random.shuffle(training_input)
            print "Starting learning epoch %d of %d" % (epoch+1, epochs)
            for batch in range((len(training_input)/mini_batch_size)):
                self.update_mini_batch(training_input, mini_batch_size, batch, learning_factor, lmbda)
            print "Finished learning epoch %d" % (epoch+1)
            if monitor_test_accuracy:
                epoch_test_accuracy_result = self.evaluate(test_input)
                print "Epoch %d test accuracy: %d / %d" % (epoch+1, epoch_test_accuracy_result, len(test_input))
                average_test_accuracy.append(epoch_test_accuracy_result / float(len(test_input)))
                if epoch_test_accuracy_result > best_results[0]:
                    best_results[0] = epoch_test_accuracy_result
                    best_epochs[0] = epoch+1
                    if save_file:
                        save_matrices.save_matrices(save_file, self.weights, self.biases, self.num_layers, epoch)
            if monitor_test_cost:
                epoch_test_cost_result = self.total_cost(test_input, lmbda, convert=True)
                    # [self.cost.cost(self.feed_forward(inp[0]), mnist_loader.vectorized_result(inp[1]))
                    #  for inp in test_input]) / len(test_input)
                print "Epoch %d average test cost: %f" % (epoch+1, epoch_test_cost_result)
                average_test_cost.append(epoch_test_cost_result)
                if epoch_test_cost_result < best_results[1]:
                    best_results[1] = epoch_test_cost_result
                    best_epochs[1] = epoch+1
            if monitor_training_accuracy:
                epoch_training_accuracy_result = self.evaluate(training_input, training_data=True)
                print "Epoch %d training accuracy: %d / %d" % (epoch+1, epoch_training_accuracy_result, len(training_input))
                average_training_accuracy.append(epoch_training_accuracy_result / len(training_input))
                if epoch_training_accuracy_result > best_results[2]:
                    best_results[2] = epoch_training_accuracy_result
                    best_epochs[2] = epoch+1
            if monitor_training_cost:
                epoch_training_cost_result = self.total_cost(training_input, lmbda)
                    # [self.cost.cost(self.feed_forward(inp[0]), inp[1])
                    #  for inp in training_input]) / len(training_input)
                print "Epoch %d average training cost: %f" % (epoch+1, epoch_training_cost_result)
                average_training_cost.append(epoch_training_cost_result)
                if epoch_training_cost_result < best_results[3]:
                    best_results[3] = epoch_training_cost_result
                    best_epochs[3] = epoch+1
        if monitor_test_accuracy:
            print "Best epoch for test accuracy: %d" % (best_epochs[0])
            print "Best result: %d / %d" % (best_results[0], len(test_input))
            ofit.plot_test_accuracy(average_test_accuracy, epochs, 0)# epochs/2)
            # plot_graph(average_test_accuracy, test_accuracy=True)
        if monitor_test_cost:
            print "Best epoch for average test cost: %d" % (best_epochs[1])
            print "Best result: %f" % (best_results[1])
            ofit.plot_test_cost(average_test_cost, epochs, 0)
            # plot_graph(average_test_cost, test_cost=True)
        if monitor_training_accuracy:
            print "Best epoch for training accuracy: %d" % (best_epochs[2])
            print "Best result: %d / %d" % (best_results[2], len(training_input))
            ofit.plot_training_accuracy(average_training_accuracy, epochs, 0, len(training_input))
            # plot_graph(average_training_accuracy, training_accuracy=True)
        if monitor_training_cost:
            print "Best epoch for average training cost: %d" % (best_epochs[3])
            print "Best result: %f" % (best_results[3])
            ofit.plot_training_cost(average_training_cost, epochs, 0)# epochs/2)
            # plot_graph(average_training_cost, training_cost=True)

    def update_mini_batch(self, training_input, mini_batch_size, batch, learning_factor, lmbda):
        batch_inputs = np.column_stack([training_input[x][0] for x in
                                        range(batch*mini_batch_size, (batch+1)*mini_batch_size)])
        batch_activations = np.column_stack([training_input[x][1] for x in
                                             range(batch*mini_batch_size, (batch+1)*mini_batch_size)])
        [batch_b_nabla, batch_w_nabla] = self.back_propagation(batch_inputs, batch_activations)
        for layer in range(self.num_layers - 1):
            self.biases[layer] = self.biases[layer] - (learning_factor / mini_batch_size) * batch_b_nabla[layer].sum(axis=1).reshape(self.sizes[layer+1], 1)
            self.weights[layer] = (1-learning_factor*(lmbda/len(training_input)))*self.weights[layer] - (learning_factor/mini_batch_size)*batch_w_nabla[layer]

    def back_propagation(self, batch_inputs, batch_activations):
        # TODO check if deques work faster here

        # start = time.time()
        delta = []
        b_nabla = []
        w_nabla = []
        w_nabla_sum = 0
        weighted_inputs = []
        prev_activations = batch_inputs
        activations = [prev_activations]
        for layer in range(self.num_layers - 1):
            weighted_inputs_for_current_layer = np.dot(self.weights[layer], prev_activations) + self.biases[layer]
            activations_for_current_layer = sigmoid_function(weighted_inputs_for_current_layer)
            prev_activations = activations_for_current_layer
            activations.append(activations_for_current_layer)
            weighted_inputs.append(weighted_inputs_for_current_layer)
        # print "Elapsed time: %f seconds" % (time.time() - start)
        for layer in reversed(range(1, self.num_layers)):
            if layer == self.num_layers - 1:
                prev_delta = self.cost.delta(activations[layer], batch_activations, weighted_inputs[layer-1])
            else:
                prev_delta = np.multiply(np.dot(np.transpose(self.weights[layer]), prev_delta),
                                         derived_sigmoid_function(weighted_inputs[layer-1]))
            delta.insert(0, prev_delta)
            b_nabla.insert(0, prev_delta)
            # w_nabla_sum += np.column_stack([np.transpose(activations[layer-1][x])*prev_delta[x] for x in range(np.shape(batch_inputs)[1])])
            w_nabla.insert(0, np.dot(prev_delta, np.transpose(activations[layer-1])))

        # print "Elapsed time: %f seconds" % (time.time() - start)
        return b_nabla, w_nabla

    def evaluate(self, evaluation_data, training_data=False):
        test_score = 0.0
        # start = time.time()
        for data in evaluation_data:
            activation = self.feed_forward(data[0])
            if training_data:
                test_score += np.argmax(activation, 0) == np.argmax(data[1])
            else:
                test_score += np.argmax(activation, 0) == data[1]
        # print "Elapsed time: %f seconds" % (time.time() - start)
        return test_score

    def feed_forward(self, curr_activation):
        for layer in range(self.num_layers - 1):
            curr_activation = sigmoid_function(np.dot(self.weights[layer], curr_activation) + self.biases[layer])
        return curr_activation

    def total_cost(self, data, lmbda, convert=False):
        cost = 0.0
        for x, y in data:
            a = self.feed_forward(x)
            # my_cost = nn.cross_entropy_cost_function(a,y)/len(data)
            if convert: y = mnist_loader.vectorized_result(y)
            cost += self.cost.cost(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost

def sigmoid_function(x):
    # calculates the sigmoid function for all neurons
    y = 1.0/(1.0 + np.exp(-x))
    return y


def derived_sigmoid_function(x):
    # calculates the derived sigmoid function for all neurons
    y = (np.exp(-x))/((1 + np.exp(-x)) ** 2)
    return y


def plot_graph(data, test_accuracy=False,
               test_cost=False,
               training_accuracy=False,
               training_cost=False):
    if test_accuracy:
        fig_name = "Test accuracy"
        axis_name = "Accuracy"
    elif test_cost:
        fig_name = "Test cost"
        axis_name = "Cost"
    elif training_accuracy:
        fig_name = "Training accuracy"
        axis_name = "Accuracy"
    elif training_cost:
        fig_name = "Training cost"
        axis_name = "Cost"

    fig = plt.figure()
    plt.plot(range(len(data)), data)
    #plt.axis([0, epochs, 0, 1])
    plt.title(fig_name + ' over time')
    plt.ylabel(axis_name)
    plt.xlabel('Epoch')
    plt.show(block=False)
    # fig.savefig(fig_name + '.png')


def main():
    random.seed(12345678)
    np.random.seed(12345678)
    training_input, validation_input, test_input = mnist_loader.load_data_wrapper()
    start = time.time()
    net = NeuralNetwork([784, 100, 10], cost=CrossEntropyCost)
    net.gradient_descent(training_input, 3, 10, 0.1, lmbda=5.0, test_input=test_input,
                         monitor_test_accuracy=True, monitor_test_cost=True,
                         monitor_training_accuracy=True, monitor_training_cost=True)  #  'best_weights_and_biases.txt')
    print "Elapsed time: %d minutes and %d seconds" % (np.floor((time.time()-start)/60.0),
                                                       np.floor(np.mod(time.time()-start, 60)))

if __name__ == '__main__':
    main()
