from implementation.network import Network, Dense, Sigmoid
from load_data import load_retinopathy_data, load_from_pickle
import numpy as np


def load_data():
    (X_train, y_train), (X_test, y_test), shape = load_from_pickle()
    y_train_model = np.zeros((len(y_train), 1))
    for idx, data in np.ndenumerate(y_train):
        y_train_model[idx] = data
    y_test_model = np.zeros((len(y_test), 1))
    for idx, data in np.ndenumerate(y_test):
        y_test_model[idx] = data
    return (X_train, y_train_model), (X_test, y_test_model), shape


def learn():
    print('Learning')
    (X_train, y_train), (X_test, y_test), shape = load_data()
    num_pixels = 7

    network = Network()
    network.add_layer(Dense(7, Sigmoid(), input_num=num_pixels))
    network.add_layer(Dense(5, Sigmoid()))
    network.add_layer(Dense(1, Sigmoid()))
    network.fit(X_train, y_train, 400, verbose=False)
    network.save_model()
    validate()
    network.show_loss()


def validate():
    print('Validation')
    (X_train, y_train), (X_test, y_test), shape = load_data()
    network = Network.load_model()
    result = network.predict(X_test)
    print(network.calculate_loss(result,y_test))
    acc = 0
    for input, output in zip(y_test, result):
        if input == output:
           acc += 1
        print(input, ' # ',output)
    acc /= len(y_test)
    print(acc)


if __name__ == '__main__':
    learn()
