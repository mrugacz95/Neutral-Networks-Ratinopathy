import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn import datasets

from implementation.network import Network
from implementation.layer import *

from implementation.activation import *


def generate_data():
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    model = np.zeros((len(y),1))
    for idx, data in np.ndenumerate(y):
        model[idx] = data
    return X, model


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
    network = Network(0.1)
    network.add_layer(Dense(5, Sigmoid(), input_num=2))
    network.add_layer(Dense(3, Sigmoid()))
    network.add_layer(Dense(1, Sigmoid()))
    X, y = generate_data()
    network.fit(X, y, 2000)
    print(network.predict(X))
    plot_decision_boundary(network.predict, X, y)
    network.show_loss()


if __name__ == '__main__':
    main()
