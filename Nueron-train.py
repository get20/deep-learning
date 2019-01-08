import numpy as np

def train(x, y, eta, epochs):

    # initialize weights to 0
    weights = np.zeros(1 + x.shape[1])
    errors = []
    for i in range(epochs):
        error = 0
        for xi, target in zip(x, y):
            update = eta * (target - predict(xi, weights))
            # update input weights
            weights[1:] +=  update * xi
            # update bias weight
            weights[0] +=  update
            error += int(update != 0.0)
        errors.append(error)
    return weights,errors

def inputs(x, weights):
    return np.dot(x, weights[1:]) + weights[0]

# Forward propagation
def predict(x, weights):
    return np.where(inputs(x, weights) >= 0.0, 1, -1)