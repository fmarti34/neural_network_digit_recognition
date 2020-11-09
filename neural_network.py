import numpy as np


class NeuralNetwork:
    def __init__(self, sizes):
        self.sizes = sizes
        self.bias = self.initialize_bias()
        self.weights = self.initialize_weights()
        self.learning_rate = 0.9
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

    def update_weights(self):
        for i in range(len(self.nodes) - 1, 0, -1):
            self.weights[i-1] += self.learning_rate * self.error_nodes[i] * self.sigmoid_derivative(self.nodes[i]) \
                                 * self.nodes[i-1].T

    def back_propagation(self, prediction, target):
        self.error_nodes[len(self.weights)] = layer = self.squared_error_cost(prediction, target)

        for i in range(len(self.weights)-1, -1, -1):
            layer = np.dot(self.weights[i].T, layer)
            self.error_nodes[i] = layer

        self.update_weights()

    def train(self, data, labels, iterations=1):
        for _ in range(iterations):
            for i in range(len(data)):
                prediction = self.forward_pass(data[i])

                # initialize a 10 by 1 matrix of the desired output
                target = np.zeros([10, 1], dtype=int)
                target[labels[i]] = 1

                self.back_propagation(prediction, target)
                # print(self.nodes)



