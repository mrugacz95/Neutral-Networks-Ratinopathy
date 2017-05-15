import json
from sklearn import datasets

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

    def predict(self, data: np.ndarray):
        for layer in self.layers:
            data = layer.process_data(data)
        return data


def generate_data():
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y

if __name__ == '__main__':
    network = Network()
    network.add_layer(InputLayer(2))
    network.add_layer(Dense(Tanh(), 3))
    network.add_layer(Dense(Softmax(), 1))
    X, y = generate_data()
    plot_decision_boundary(lambda x: network.predict(X), X, y)
    plt.title("Decision Boundary for hidden layer size 3")
    plt.show()
