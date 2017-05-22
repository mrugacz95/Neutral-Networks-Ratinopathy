from abc import abstractmethod

import numpy as np
from scipy.special import expit


class Activation:
    @abstractmethod
    def forward(self, x: np.ndarray):
        return

    @abstractmethod
    def derivative(self, x: np.ndarray):
        return

    def to_json(self):
        return self.__str__()

    @staticmethod
    def from_json(activation:str) -> 'Activation':
        return {
            Sigmoid().__str__() : Sigmoid(),
            Tanh().__str__() : Tanh(),
            ReLU().__str__() : ReLU(),
            Softmax().__str__() : Softmax(),
        }[activation]


class Sigmoid(Activation):
    def __str__(self):
        return 'sigmoid'

    def forward(self, x: np.ndarray):
        # sig = np.vectorize(lambda: 1 / (1 + math.exp(-x))) # slow
        # return sig(x)
        return expit(x)

    def derivative(self, x: np.ndarray):
        return x * (1.0 - x)


class Tanh(Activation):
    def __str__(self):
        return 'tanh'

    def forward(self, x: np.ndarray):
        return np.tanh(x)

    def derivative(self, x: np.ndarray):
        return 1 - np.power(x, 2)


class ReLU:
    def __str__(self):
        return 'relu'

    def forward(self, x: np.ndarray):
        return np.max(x)

    def derivative(self, x: np.ndarray):  # todo pass bias
        d = np.vectorize(lambda var: 1 if var > 0 else 0)
        return d(x)


class Softmax:
    def __str__(self):
        return 'softmax'

    def forward(self, x: np.ndarray):
        exp_scores = np.exp(x)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def derivative(self, x: np.ndarray):
        pass
