from implementation.activation import Activation
import numpy as np


class Layer:
    def __init__(self, nodes_num: int):
        self.nodes_num = nodes_num


class Dense(Layer):
    def __init__(self, nodes_num: int, activation, input_shape: tuple = None,
                 weights: np.ndarray = None, bias : np.ndarray = None):
        super().__init__(nodes_num)
        self.activation = activation
        self.weights = weights
        if bias is None:
            bias = np.zeros((1, nodes_num))
        self.bias = bias
        self.error = 0
        self.top_layer = None
        self.input_shape = input_shape
        self.id = None

    def set_top_layer(self, layer: Layer):
        self.top_layer = layer
        top_size = layer.nodes_num
        if self.weights is None and self.input_shape is None:
            self.weights = np.random.randn(top_size, self.nodes_num) / np.sqrt(self.nodes_num)  # normal dist / sqrt(node_num)

    def process_data(self, data : np.ndarray):
        data = np.dot(data, self.weights) + self.bias
        data = self.activation.forward(data)
        return data

    def update_single_weight(self, delta: float, learning_rate: float):
        error = np.multiply(self.weights, delta)
        dev = self.activation.derivative(error)
        self.weights = self.weights * learning_rate * dev
        return np.sum(error)

    def set_id(self, id):
        self.id = id

    def __str__(self):
        return str(self.id) + " weights:\n" + str(self.weights) + "\nbias: " + str(self.bias)
