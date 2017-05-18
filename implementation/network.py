import json

from sklearn.metrics import mean_squared_error

from implementation.activation import *
from implementation.layer import *
import numpy as np
import matplotlib.pyplot as plt


class Network:
    def __init__(self):
        self.layers = list()

    def load_model(self, model_path):
        with open(model_path)as model_file:
            data = json.load(model_file)
            # todo parse model

    def save_model(self):
        # todo parse model
        pass

    def add_layer(self, layer: Layer):
        if isinstance(layer, Dense):
            layer.set_bottom_layer(self.layers[-1])
        self.layers.append(layer)

    def forward_propagation(self, data: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            data = layer.process_data(data)
        return data

    def backward_propagation(self, loss: float):
        pass

    def predict(self, data: np.ndarray) -> np.ndarray:
        data = self.forward_propagation(data)
        return np.argmax(data, axis=1)

    def calculate_error(self, output: np.ndarray, model: np.ndarray) -> np.ndarray:
        return np.sum(((model - output) / 2) ** 2)

