from implementation.activation import Activation
import numpy as np


class Layer:
    def __init__(self, nodes_num: int):
        self.nodes_num = nodes_num


class Dense(Layer):
    def __init__(self, activation, nodes_num: int):
        super().__init__(nodes_num)
        self.activation = activation
        self.bottom_layer = None
        self.weights = None
        self.bias = np.zeros((1, nodes_num))
        self.error = 0

    def set_bottom_layer(self, layer):
        self.bottom_layer = layer
        bottom_size = layer.nodes_num
        self.weights = np.random.randn(bottom_size, self.nodes_num) / np.sqrt(self.nodes_num)  # normal dist / sqrt

    def process_data(self, data):
        data = np.dot(data, self.weights) + self.bias
        data = self.activation.forward(data)
        return data

    def update_single_weight(self, delta: float, learning_rate: float):
        error = np.multiply(self.weights, delta)
        dev = self.activation.derivative(error)
        self.weights = self.weights * learning_rate * dev
        return np.sum(error)


class InputLayer(Layer):
    def __init__(self, nodes_num: int):
        super().__init__(nodes_num)
        self.nodes_num = nodes_num

    def process_data(self, data: np.ndarray):
        if not data.shape != (None, self.nodes_num):
            raise NameError('Wrong input size')
        return data
