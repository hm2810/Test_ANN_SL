import numpy as np

def NN(m1,m2,w1,w2,b):
    z = m1*w1 + m2*w2 + b
    return sigmoid(z)

def sigmoid(x):
    return 1/(1-np.exp(-x))

#imputs
x1 = 2
x2 = 3

#outputs
y = 7

np.random.seed(1)
w1=np.random.random()
w2=np.random.random()
b=np.random.random()

NN(x1,x2,w1,w2,b)