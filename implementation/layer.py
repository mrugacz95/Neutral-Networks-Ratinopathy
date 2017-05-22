from abc import abstractmethod

from implementation.activation import Activation
import numpy as np


class Layer:
    id = None

    def __init__(self, nodes_num: int):
        self.nodes_num = nodes_num

    def set_id(self, id: int):
        self.id = id

    @abstractmethod
    def set_bottom_layer(self, layer: 'Layer'):
        pass

    @staticmethod
    def from_json(json_obj):
        layer = {
            'dense': Dense
        }[json_obj['type']]
        return layer(nodes_num=json_obj['nodes_num'], activation=Activation.from_json(json_obj['activation']),
                     input_num=json_obj['input_num'], weights=np.array(json_obj['weights']), bias=np.array(json_obj['bias']))


class Dense(Layer):
    data = None
    output = None

    def __init__(self, nodes_num: int, activation: Activation, input_num: int = None,
                 weights: np.ndarray = None, bias: np.ndarray = None):
        super().__init__(nodes_num)
        self.activation = activation
        self.weights = weights
        if bias is None:
            bias = np.zeros(nodes_num)
        self.bias = bias
        self.error = 0
        self.bottom_layer = None
        self.input_num = input_num
        if self.weights is None and self.input_num is not None:
            self.init_weights()

    def set_bottom_layer(self, layer: Layer):
        self.input_num = layer.nodes_num
        if self.weights is None:
            self.init_weights()

    def process_data(self, data: np.ndarray):
        result = np.dot(data, self.weights) + self.bias
        result = self.activation.forward(x=result)
        self.output = result
        self.data = data
        return result

    def update_single_weight(self, delta: np.ndarray, learning_rate: float, data_idx: int):
        derivative = self.activation.derivative(self.output[data_idx])
        next_error = np.sum(np.multiply(delta.T, self.weights) * derivative, axis=1)
        my_delta = np.multiply(derivative, delta)
        my_delta *= learning_rate
        self.bias += my_delta
        for j in range(self.nodes_num):
            for i in range(self.input_num):
                self.weights[i][j] += my_delta[j] * self.data[data_idx][i]
        return next_error

    def init_weights(self):
        self.weights = np.random.randn(self.input_num,
                                       self.nodes_num)  # / np.sqrt(self.nodes_num)  # normal dist / sqrt(node_num)

    def print(self):
        return "layer no: " + str(self.id) + "\nweights:\n" + str(self.weights) + "\nbias: " + str(self.bias)

    def __str__(self):
        return 'dense'

    def to_json(self):
        return {'type': self.__str__(), 'nodes_num': self.nodes_num, 'input_num': self.input_num, 'id': self.id,
                'weights': self.weights.tolist(), 'bias': self.bias.tolist(),
                'activation': self.activation.to_json()}
