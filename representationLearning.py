import numpy as np
import scipy as sp
import scipy.optimize as opt

def generateA():
     A = np.random.randn(20)
     A = A.reshape((10,2))
     return A

def generateX():
    x = np.linspace(0,4*np.pi,100)
    X = np.zeros((100,2))
    X[:,0]= np.cos(x)
    X[:,1]=np.sin(x)
    return(X)

def generateY():
    A = generateA()
    X = generateX()
    Y = np.dot(X, np.transpose(A))
    return Y


def f(W, Y):
    H = np.dot(W,np.transpose(W))+ np.ones()
    B = np.dot()


print(np.shape(generateY()))