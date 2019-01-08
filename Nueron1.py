import numpy as np
#set inputs
inputs=np.array([5,1,4,8])

#set weights
weights=np.array([2,-1,3,-2])
#Define a simple activation function
def f(x):
    return x
#compute output
output=f(sum(weights.T * inputs))
print(output)