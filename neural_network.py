import numpy as np
from tqdm.auto import trange
import os
import pickle

DIR = os.getcwd()


class NeuralNetwork:
    def __init__(self, sizes, train=False):
        """
        :param sizes: list
            A list of integers used to determine the sizes of the input, hidden, and output layers.
            ex: [784, 128, 10] : Will initialize a 784x128 matrix of weights and bias. Also, it will initialize a
                                 128x10 matrix of weights and bias.
        :param train: boolean, optional
            When this flag is true it will randomize the biases and weights even if it can load model from disk
        """
        self.sizes = sizes
        self.learning_rate = 0.01
        self.nodes = {}
        self.error_nodes = {}
        self.trained = False

        # set biases and weights to random number if not able to load model from disk
        if train or self.load() is False:
            self.biases = self.initialize_bias()
            self.weights = self.initialize_weights()
        else:
            self.trained = True

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def squared_error_cost(prediction, target, derivative=False):
        if derivative:
            return 2 * (prediction - target)
        return (prediction - target) ** 2

    def activation(self, inputs, weights, bias):
        """
        Calculates the next layer by using the dot product, it multiplies the weights matrix by the inputs matrix
        connecting them to the neurons in the next layer. Next it adds the bias vector using matrix addition. Lastly,
        it applies the sigmoid function to the layer.

        :param inputs: numpy.ndarray
        :param weights: numpy.ndarray
        :param bias: numpy.ndarray
        :return: numpy.ndarray
        """
        return self.sigmoid(np.dot(weights, inputs) + bias)

    def sigmoid_derivative(self, x):
        """
        return self.sigmoid(x) * (1 - self.sigmoid(x)) - actual sigmoid derivative

        since sigmoid is already applied to x in forward_pass() we do not apply sigmoid to x again

        :param x: numpy.ndarray
        :return: numpy.ndarray
        """
        return x * (1 - x)

    def initialize_weights(self):
        """
        Initializes random matrices that will be used as weights. Each weight matrix is added to dictionary with the
        corresponding index as its key.

        :return: dict
        """
        weights = {}
        for i in range(1, len(self.sizes)):
            weights[i-1] = np.random.randn(self.sizes[i], self.sizes[i-1]) * np.sqrt(1. / self.sizes[i])
        return weights

    def initialize_bias(self):
        """
        Initializes random matrices that will be used as biases. Each bias matrix is added to dictionary with the
        corresponding index as its key.

        :return: dict
        """

        bias = {}
        for i in range(1, len(self.sizes)):
            bias[i-1] = np.random.randn(self.sizes[i], 1) * np.sqrt(1. / self.sizes[i])
        return bias

    def save(self):
        """
        save weights and biases to disk in '/model' folder
        """
        try:
            with open(DIR + '/model/weights.pickle', 'wb') as handle:
                pickle.dump(self.weights, handle)

            with open(DIR + '/model/bias.pickle', 'wb') as handle:
                pickle.dump(self.biases, handle)
        except FileNotFoundError as e:
            print('ERROR: Could not save files successfully: ' + e)



    def load(self):
        """
            Loads weights and biases from disk
            :return: Boolean
        """
        try:
            with open(DIR + '/model/weights.pickle', 'rb') as handle:
                self.weights = pickle.load(handle)

            with open(DIR + '/model/bias.pickle', 'rb') as handle:
                self.biases = pickle.load(handle)
        except IOError:
            return False

        return True

    def forward_pass(self, image):
        """
        Calculates every layer and stores it in self.nodes dictionary

        :param image: numpy.ndarray
        :return: numpy.ndarray
            This is the output layer
        """

        self.nodes[0] = layer = image.reshape((28, 28)).reshape(28 ** 2, 1)  # convert image to 784x1 matrix

        for i in range(len(self.weights)):
            layer = self.activation(layer, self.weights[i], self.biases[i])
            self.nodes[i+1] = layer

        return layer

    def predict_image(self, image):
        """
        Predicts the handwritten digit that's in the image

        :param image: numpy.ndarray
        :return: numpy.int64
        """
        return np.argmax(self.forward_pass(image), axis=0)[0]

    def update_weights_iter(self, target):

        index = len(self.weights) - 1
        # print(index)
        self.weights[index] += self.learning_rate * (target - self.nodes[index+1]).dot(self.nodes[index].T)
        self.biases[index] += self.learning_rate * (target - self.nodes[index + 1]).sum()

        # self.weights[2] += self.learning_rate * (target - self.nodes[3]).dot(self.nodes[2].T)
        # self.bias[2] += self.learning_rate * (target - self.nodes[3]).sum()

        for i in range(len(self.weights)-2, -1, -1):
            # print(i)
            gradient = self.weights[i+1].T.dot(2*(self.error_nodes[i+2] - self.nodes[i+2])) * self.sigmoid_derivative(self.nodes[i+1])
            self.weights[i] += self.learning_rate * gradient.dot(self.nodes[i].T)
            self.biases[i] += self.learning_rate * gradient

        # gradient = self.weights[2].T.dot(target - self.nodes[3]) * self.sigmoid_derivative(self.nodes[2])
        # self.weights[1] += self.learning_rate * gradient.dot(self.nodes[1].T)
        # self.bias[1] += self.learning_rate * gradient

        # gradient = self.weights[1].T.dot(self.error_nodes[2]) * self.sigmoid_derivative(self.nodes[1])
        # self.weights[0] += self.learning_rate * gradient.dot(self.nodes[0].T)
        # self.bias[0] += self.learning_rate * gradient

    def update_weights(self, target):
        self.weights[2] += self.learning_rate * (target - self.nodes[3]).dot(self.nodes[2].T)
        self.biases[2] += self.learning_rate * (target - self.nodes[3]).sum()

        gradient = self.weights[2].T.dot(target - self.nodes[3]) * self.sigmoid_derivative(self.nodes[2])
        self.weights[1] += self.learning_rate * gradient.dot(self.nodes[1].T)
        self.biases[1] += self.learning_rate * gradient

        gradient = self.weights[1].T.dot(self.error_nodes[2]) * self.sigmoid_derivative(self.nodes[1])
        self.weights[0] += self.learning_rate * gradient.dot(self.nodes[0].T)
        self.biases[0] += self.learning_rate * gradient

    def update_bias(self):
        for i in range(len(self.nodes) - 1, 0, -1):
            self.weights[i-1] += self.learning_rate * self.error_nodes[i] * self.sigmoid_derivative(self.nodes[i]) \
                                 * self.nodes[i-1].T

    def back_propagation(self, prediction, target):
        self.error_nodes[len(self.weights)] = layer = self.squared_error_cost(prediction, target)

        for i in range(len(self.weights)-1, -1, -1):
            layer = np.dot(self.weights[i].T, layer)
            self.error_nodes[i] = layer

        self.update_weights(target)

    def train(self, data, labels, iterations=5):
        for iter in range(iterations):
            for i in trange(len(data),  desc=str(iter) + '/' + str(iterations)):
                prediction = self.forward_pass(data[i])

                # if i % 10 == 0:
                #     self.predict_image(data[i])
                #     time.sleep(1)

                # initialize a 10 by 1 matrix of the desired output
                target = np.zeros([10, 1], dtype=int)
                target[labels[i]] = 1


                self.back_propagation(prediction, target)
                # print(self.nodes)
        self.save()
