from sklearn import datasets

from implementation.network import Network, Tanh, Dense, Sigmoid
import numpy as np


def main():
    network = Network(learning_rate=0.5)
    network.add_layer(
        Dense(2, Sigmoid(), input_shape=(2,), weights=np.array([[.15, 0.25], [.2, .3]]),
              bias=np.array([.35, .35])))
    network.add_layer(
        Dense(2, Sigmoid(), input_shape=(2,), weights=np.array([[.4, 0.5], [.45, .55]]), bias=np.array([.6, .6])))
    network.print_weights()
    output = network.forward_propagation(np.array([[.05, .1]]))
    error = network.calculate_error(output, np.array([0.01,0.99]))
    print(error)
    print('output:',output)
    network.online_backward_propagation(error)
    network.print_weights()


if __name__ == '__main__':
    main()
