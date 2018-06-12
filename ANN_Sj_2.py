import numpy as np

#sigmoid
def sigmoid(x):
     return 1/(1+np.exp(-x))

#derivative
def derivative(x):
    return (x*(1-x))

#input for training
x = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

#seed
np.random.seed(1)

#weights and bias
b = 1
w0 = 2*np.random.random((3,4)) - b
w1 = 2*np.random.random((4,1)) - b

print('Initial weight w0:')
print(w0)
print('Initial weight w1:')
print(w1)

#training
for i in range(10000):
    #layers
    l0 = x
    l1 = sigmoid(np.dot(l0,w0))
    l2 = sigmoid(np.dot(l1,w1))

    #backpropagation
    l2_error = y -l2
    
    #print error
    if (i%1000) == 0:
        print('Error after '+ str(i) + ' is:' + str(np.mean(np.abs(l2_error))))
    
    #calculate deltas
    l2_delta = l2_error * derivative(l2)
    l1_error = np.dot(l2_delta, w1.T)
    l1_delta = l1_error * derivative(l1)
    
    #update weights
    w1 += np.dot(l1.T,l2_delta)
    w0 += np.dot(l0.T, l1_delta)
    
print('Predicted result is')    
print (l2)
print('Final weight w0:')
print(w0)
print('Final weight w1:')
print(w1)