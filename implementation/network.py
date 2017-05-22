import json
import sys
from implementation.activation import *
from implementation.layer import *
import numpy as np
import matplotlib.pyplot as plt
from etaprogress.progress import ProgressBar


class Network:
    def __init__(self, learning_rate=0.01):
        self.layers = list()
        self.learning_rate = learning_rate
        self.history = None

    @staticmethod
    def load_model(model_path: str = 'my_model.json') -> 'Network':
        with open(model_path) as model_file:
            json_obj = json.load(model_file)
            return Network.from_json(json_obj)

    def save_model(self, model_path: str = 'my_model.json'):
        with open(model_path, 'w+') as outfile:
            json.dump(self.to_json(), outfile)
        pass

    def add_layer(self, layer: Layer):
        layer.set_id(len(self.layers))
        if self.layers:  # not empty
            layer.set_bottom_layer(self.layers[-1])
        self.layers.append(layer)

    def forward_propagation(self, data: np.ndarray) -> np.ndarray:
        if data.shape[1] != self.layers[0].input_num:
            raise NameError("Wrong dataset size", data.shape[1], 'and', self.layers[0].input_num)
        for layer in self.layers:
            data = layer.process_data(data)
        return data

    def online_backward_propagation(self, error: np.ndarray):
        for idx in range(len(error)):
            current_error = error[idx]
            for layer in reversed(self.layers):
                current_error = layer.update_single_weight(current_error, self.learning_rate, idx)

    def predict(self, data: np.ndarray) -> np.ndarray:
        data = self.forward_propagation(data)
        return data

    def calculate_loss(self, output: np.ndarray, model: np.ndarray) -> np.ndarray:
        # return np.sum(self.calculate_error(output, model))/len(model)
        return np.sum(((model - output) ** 2) / 2) / len(model)

    def calculate_error(self, output: np.ndarray, model: np.ndarray) -> np.ndarray:
        error = model - output
        return error

    def fit(self, X, y, iters: int, verbose: bool = True):
        self.history = np.empty(iters)
        bar = ProgressBar(iters)
        for i in range(iters):
            result = self.forward_propagation(X)
            errors = self.calculate_error(output=result, model=y)
            self.online_backward_propagation(errors)
            loss = self.calculate_loss(output=result, model=y)
            self.history[i] = loss
            if not verbose and i % 150 == 0:
                bar.numerator = i
                sys.stdout.write('\r')
                sys.stdout.write(str(bar))
                sys.stdout.flush()
        if not verbose:
            print('Finished fitting')

    def print_weights(self):
        for layer in self.layers:
            print(layer)

    def show_loss(self):
        plt.plot(self.history)
        plt.title('model accuracy')
        plt.ylabel('loss')
        plt.xlabel('iter')
        plt.legend(['loss'], loc='upper left')
        plt.show()

    def to_json(self):
        json_layers = list()
        for layer in self.layers:
            json_layers.append(layer.to_json())
        return json_layers

    @staticmethod
    def from_json(json_object) -> 'Network':
        network = Network()
        for json_layer in json_object:
            network.add_layer(Layer.from_json(json_obj=json_layer))
        return network
