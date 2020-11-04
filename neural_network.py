import numpy as np


class NeuralNetwork:
    def __init__(self, sizes):
        self.sizes = sizes
        self.weights = self.initialize_weights()

    def neural_network(self, input1, input2, w1, w2, bias):
        return self.sigmoid(input1 * w1 + input2 * w2 + bias)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def initialize_weights(self):
        weights = {}
        for i in range(1, len(self.sizes)):
            print(self.sizes[i-1], self.sizes[i])
            weights[i-1] = np.random.randn(self.sizes[i], self.sizes[i-1]) * np.sqrt(1. / self.sizes[i])
        return weights
