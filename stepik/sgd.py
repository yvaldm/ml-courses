import numpy as np


def SGD(self, X, y, batch_size, learning_rate=0.1, eps=1e-6, max_steps=200):


# here goes your code

def update_mini_batch(self, X, y, learning_rate, eps):
    # here goes your code
    range = np.arange(0, len(X), 1)
    np.random.shuffle(range)
    self.compute_grad_analytically()
