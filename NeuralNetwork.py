import numpy as np


class NeuralNetwork:
    def __init__(self, sizes):
        self.sizes = sizes

    def neural_network(self, input1, input2, w1, w2, bias):
        return self.sigmoid(input1 * w1 + input2 * w2 + bias)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
