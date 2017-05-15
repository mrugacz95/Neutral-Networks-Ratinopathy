import numpy as np
from scipy.special import expit


class Activation():
    def forward(self, x: np.ndarray):
        pass


class Sigmoid(Activation):
    def forward(self, x: np.ndarray):
        # sig = np.vectorize(lambda: 1 / (1 + math.exp(-x)))
        # return sig(x)
        return expit(x)


class Tanh(Activation):
    def forward(self, x: np.ndarray):
        return np.tanh(x)


class ReLU(Activation):
    def forward(self, x: np.ndarray):
        return np.max(x)


class Softmax(Activation):
    def forward(self, x: np.ndarray):
        exp_scores = np.exp(x)
        probs =  exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)

