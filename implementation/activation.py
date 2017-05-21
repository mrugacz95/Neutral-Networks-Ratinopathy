from abc import abstractmethod

import numpy as np
from scipy.special import expit


class Activation:
    @abstractmethod
    def forward(self, x: np.ndarray):
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray):
        pass


class Sigmoid:
    def forward(self, x: np.ndarray):
        # sig = np.vectorize(lambda: 1 / (1 + math.exp(-x))) # slow
        # return sig(x)
        return expit(x)

    def derivative(self, x: np.ndarray):
        return x * (1.0 - x)


class Tanh:
    def forward(self, x: np.ndarray):
        return np.tanh(x)

    def derivative(self, x: np.ndarray):
        return 1 - np.power(x, 2)


class ReLU:
    def forward(self, x: np.ndarray):
        return np.max(x)

    def derivative(self, x: np.ndarray):  # todo pass bias
        d = np.vectorize(lambda x: 1 if x > 0 else 0)
        return d(x)


class Softmax:
    def forward(self, x: np.ndarray):
        exp_scores = np.exp(x)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def derivative(self, x: np.ndarray):
        return x * (1.0 - x)
