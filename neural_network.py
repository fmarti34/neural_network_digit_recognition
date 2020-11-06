import numpy as np


class NeuralNetwork:
    def __init__(self, sizes):
        self.sizes = sizes
        self.bias = np.random.random()
        self.weights = self.initialize_weights()

    def neural_network(self, inputs, weights):
        return self.sigmoid(np.dot(weights, inputs) + self.bias)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def initialize_weights(self):
        weights = {}
        for i in range(1, len(self.sizes)):
            print(self.sizes[i-1], self.sizes[i])
            weights[i-1] = np.random.randn(self.sizes[i], self.sizes[i-1]) * np.sqrt(1. / self.sizes[i])
        return weights

    def forward_pass(self, image):
        layer = image.reshape((28, 28)).reshape(28 ** 2, 1)  # convert image to 784x1 vector

        for i in range(len(self.weights)):
            layer = self.neural_network(layer, self.weights[i])

        print(layer)
        print(np.argmax(layer, axis=0)[0])
