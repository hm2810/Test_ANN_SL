import numpy as np

#sigmoid
def nonlin(x, deriv=False):
    if deriv==True:
        return (x*(1-x))
    
    return 1/(1+np.exp(-x))

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

#training
for i in range(10000):
    #layers
    l0 = x
    l1 = nonlin(np.dot(l0,w0))
    l2 = nonlin(np.dot(l1,w1))

    #backpropagation
    l2_error = y -l2
    
    #print error
    if (i%1000) == 0:
        print('Error after '+ str(i) + ' is:' + str(np.mean(np.abs(l2_error))))
    
    #calculate deltas
    l2_delta = l2_error * nonlin(l2, deriv=True)
    l1_error = np.dot(l2_delta, w1.T)
    l1_delta = l1_error * nonlin(l1, deriv = True)
    
    #update weights
    w1 += np.dot(l1.T,l2_delta)
    w0 += np.dot(l0.T, l1_delta)
    
print('Predicted result is')    
print (l2)
    
    
    
    
    