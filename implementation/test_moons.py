import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn import datasets

from implementation.network import Network
from implementation.layer import *

from implementation.activation import *


def pack_targets(data):
    targets = np.zeros((len(data), 1))
    for idx, data in np.ndenumerate(data):
        targets[idx] = data
    return targets


def generate_moons(examples: int, noise: float = 0.2):
    X, y = datasets.make_moons(examples, noise=noise)
    return X, pack_targets(y)


def generate_circles(examples: int, noise: float = 0.2, factor: float = 0.3):
    X, y = datasets.make_circles(examples, noise=noise, factor=factor)
    return X, pack_targets(y)


def generate_classification(examples: int = 250):
    X, y = datasets.make_classification(examples, n_features=2, n_redundant=0)
    return X, pack_targets(y)


def generate_random():
    return {
        0: generate_moons,
        1: generate_circles,
        2: generate_classification,

    }[random.randint(0, 2)]


def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = np.reshape(Z, xx.shape)
    plt.contourf(xx, yy, Z)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm.gray)
    plt.show()


def main():
    network = Network(learning_rate=.1)
    network.add_layer(Dense(nodes_num=5, activation=Tanh(), input_num=2))
    network.add_layer(Dense(nodes_num=1, activation=Sigmoid()))
    inputs, targets = generate_circles(400)
    # fit model
    network.fit(inputs, targets, 380, verbose=True)
    # learn
    network.show_loss()
    network.save_model()
    plot_decision_boundary(network.predict, inputs, targets)


if __name__ == '__main__':
    main()
