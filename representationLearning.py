import numpy as np
import scipy as sp
import scipy.optimize as opt
import matplotlib.pyplot as plt

def generateA():
     A = np.random.randn(20)
     A = A.reshape((10,2))
     return A

def generateX():
    x = np.linspace(0,4*np.pi,100)
    X = np.zeros((100,2))
    X[:,0]= np.multiply(x,np.cos(x))
    X[:,1]= np.multiply(x,np.sin(x))
    return(X)

def generateY():
    A = generateA()
    X = generateX()
    Y = np.dot(X, np.transpose(A))
    noise = np.random.multivariate_normal(np.zeros(10), 0.1*np.eye(10), 100)
    return Y+noise

sigma = 2
Y = generateY()

def f(W):
    W = np.reshape(W,(10,2))
    H = np.dot(W,np.transpose(W))
    I = sigma*np.eye(10)
    inv = np.linalg.inv(H+I)
    A = 50*np.log(np.linalg.det(H+I))
    B = 0.5*np.trace(np.dot(inv, np.dot(np.transpose(Y),Y)))
    return A+B+0.5*10*100*np.log(2*np.pi)

def df(W):
    W = np.reshape(W,(10,2))
    H = np.dot(W,np.transpose(W))
    I = sigma*np.eye(10)
    inv = np.linalg.inv((H+I))

    val = np.empty(W.shape)
    for i in range(val.shape[0]):
        for j in range(val.shape[1]):
            J = np.zeros(np.shape(W))
            J[i,j] = 1
            dWW = np.dot(J,np.transpose(W)) + np.dot(W,np.transpose(J))
            A = 100*np.trace(np.dot(inv,dWW))
            B1 = np.dot(np.dot(-inv, dWW),inv)
            B = np.trace(np.dot(np.dot(np.transpose(Y),Y), B1))
            val[i,j]= 0.5*A+0.5*B
    val = np.reshape(val,(20,))
    return val

A = 20*np.random.randn(20)
A = np.reshape(A, (20,))
W0 = np.ones(20)
Wstar = opt.fmin_cg(f,A, fprime=df)
W = np.reshape(Wstar,(10,2))
WtW = np.dot(np.transpose(W),W)
inv = np.linalg.pinv(WtW)
X = np.dot(Y, np.dot(W,WtW))
plt.scatter(X[:,0],X[:,1])
plt.show()


