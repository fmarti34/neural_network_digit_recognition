import numpy as np
from tqdm.auto import trange
import time


class NeuralNetwork:
    def __init__(self, sizes):
        self.sizes = sizes
        self.bias = self.initialize_bias()
        self.weights = self.initialize_weights()
        self.learning_rate = 0.001
        self.nodes = {}
        self.error_nodes = {}

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def squared_error_cost(prediction, target, derivative=False):
        if derivative:
            return 2 * (prediction - target)
        return (prediction - target) ** 2

    def activation(self, inputs, weights, bias):
        return self.sigmoid(np.dot(weights, inputs) + bias)

    def sigmoid_derivative(self, x):
        # return self.sigmoid(x) * (1 - self.sigmoid(x)) - actual sigmoid derivative
        # since sigmoid is already applied to x in forward_pass()
        return x * (1 - x)

    def initialize_weights(self):
        weights = {}
        for i in range(1, len(self.sizes)):
            weights[i-1] = np.random.randn(self.sizes[i], self.sizes[i-1]) * np.sqrt(1. / self.sizes[i])
        return weights

    def initialize_bias(self):
        bias = {}
        for i in range(1, len(self.sizes)):
            bias[i-1] = np.random.randn(self.sizes[i], 1) * np.sqrt(1. / self.sizes[i])
        return bias

    def forward_pass(self, image):
        self.nodes[0] = layer = image.reshape((28, 28)).reshape(28 ** 2, 1)  # convert image to 784x1 matrix

        for i in range(len(self.weights)):
            layer = self.activation(layer, self.weights[i], self.bias[i])
            self.nodes[i+1] = layer

        return layer

    def predict_image(self, image):
        return np.argmax(self.forward_pass(image), axis=0)[0]

    def update_weights(self, target):

        self.weights[2] += self.learning_rate * (target - self.nodes[3]).dot(self.nodes[2].T)
        self.bias[2] += self.learning_rate * (target - self.nodes[3]).sum()

        gradient = self.weights[2].T.dot(target - self.nodes[3]) * self.sigmoid_derivative(self.nodes[2])
        self.weights[1] += self.learning_rate * gradient.dot(self.nodes[1].T)
        self.bias[1] += self.learning_rate * gradient

        gradient = self.weights[1].T.dot(self.error_nodes[2]) * self.sigmoid_derivative(self.nodes[1])
        self.weights[0] += self.learning_rate * gradient.dot(self.nodes[0].T)
        self.bias[0] += self.learning_rate * gradient

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



