from sklearn import datasets

from implementation.network import Network, Tanh, Dense, Sigmoid
import numpy as np


def main():
    network = Network(learning_rate=0.1)
    network.add_layer(
        Dense(2, Sigmoid(), input_num=2, weights=np.array([[.1, 0.3], [.2, -0.1]]),
              bias=np.array([.2, .1])))
    network.add_layer(
        Dense(1, Sigmoid(), weights=np.array([[-0.1],[.2],]), bias=np.array([-0.2])))
    network.print_weights()
    output = network.forward_propagation(np.array([[1, 0], [1, 0], [1, 0]]))
    print('output', output)
    error = network.calculate_error(output, np.array([[1],[1],[1]]))
    print('error',error)
    error = np.array([[0.5343]])
    network.online_backward_propagation(error)
    network.print_weights()

    output = network.forward_propagation(np.array([[1, 0], [1, 0], [1, 0]]))
    print('output', output)
    print('output should be around 0.4712')


if __name__ == '__main__':
    main()
