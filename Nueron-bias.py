import numpy as np


# set inputs
inputs = np.array([5, 1, 4, 8])

# set weights
weights = np.array([2, -1, 3, -2])

bias = -2.5 

# Define a simple activation function
def f(x):
  return 1 / (1 + np.exp(-x))

# Compute output
output = f(sum(weights.T*inputs) + bias)

print(output) 