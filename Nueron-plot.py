import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
%matplotlib inline


plt.figure(figsize=(20,10))

def plot_activation(x,y,title, position):  
    ax = plt.subplot2grid((3, 3), position)                    
    ax.plot(x,y)
    if position[0]==2:
        ax.set_xlabel('Input')
    else:
        ax.set_xticklabels([])
    ax.set_title(title)
    ax.set_ylabel('Output');
                          
x = np.arange(-5, 5, 0.01)

# identity function
y=x
title='Identity'
plot_activation(x,y,title,(0,0))

# binary step
y =(x>=0)*1
title='Binary step'
plot_activation(x,y,title,(0,1))

# Logistic Activation Function
y = 1 / (1 + np.exp(-x))
title='Logistic'
plot_activation(x,y,title,(0,2))

# tanH function
y=np.tanh(x)
title='tanH'
plot_activation(x,y,title,(1,0))

# arcTan function
y = np.arctan(x)
title='ArcTan'
plot_activation(x,y,title,(1,1))

# Rectified Linear
y = (x>=0)*x
title='Rectified Linear'
plot_activation(x,y,title,(1,2))

# Parametric Rectified Linear Unit function
alpha=0.5
y = (x<0)*alpha*x + (x>=0)*x
title='PReLU'
plot_activation(x,y,title,(2,0))

# Exponential Linear Unit function
alpha=1
y = x*(x>=0) + (alpha*(np.exp(x)-1))*(x<0)
title='ELU'
plot_activation(x,y,title,(2,1))

#  Linear
y = np.log(1+np.exp(x))
title='SoftPlus'
plot_activation(x,y,title,(2,2))