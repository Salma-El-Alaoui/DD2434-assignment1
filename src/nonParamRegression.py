
_author__ = 'Salma'

import numpy as np
import pylab as pb
from math import pi
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist

def plotGp(l):
    X = np.linspace(-4.0,4.0,2000) # 500 points evenly spaced over [0,1]
    X = X[:,None] # reshape X to make it n*D
    mu = np.zeros((2000)) # vector of the means
    K =  np.exp(-cdist(X, X, 'sqeuclidean')/(l*l))# covariance matrix
    # Generate 20 sample path with mean mu and covariance C
    Z = np.random.multivariate_normal(mu,K,20)
    pb.figure() # open new plotting window
    for i in range(20):
        pb.plot(X[:],Z[i,:])
    title = 'length-scale '+str(l)
    pb.title(title)
    pb.show()

def generateData():
    X = np.array([-pi, -3*pi/4,-pi/2,0, pi/2, 3*pi/4, pi])
    epsilon = np.sqrt(0.5) * np.random.randn(7)
    Y = np.sin(X)+epsilon
    return (X,Y)

def kernel(X, Y, l=1.):
    return np.exp(-cdist(X, Y, 'sqeuclidean')/(l*l))

def computePosterior(xStar, X, Y,l):
    #Xstar = np.array([xStar])
    Xstar = xStar
    X = X[:,None]
    Xstar = Xstar[:,None]
    k = kernel(Xstar,X,l)
    C = np.linalg.inv(kernel(X,X,l))
    t = Y[:,None]
    mu = np.dot(np.dot(k,C),t)

    c = kernel(Xstar, Xstar,l)
    sigma = c- np.dot(np.dot(k,C),np.transpose(k))
    return mu, sigma

def plotSamplePos():
    X, Y = generateData()
    x = np.linspace(-2*np.pi, 2*np.pi, 800)
    mu, sigma=computePosterior(x,X,Y,2.)
    print(mu)
    mu = np.reshape(mu,(800,))
    x = x[:,None]
    Z = np.random.multivariate_normal(mu,np.nan_to_num(sigma),20)
    pb.figure() # open new plotting window?
    pb.plot(X,Y,'ro')
    for i in range(20):
        pb.plot(x[:],Z[i,:])
    pb.show()

def plotPosterior():
    #plotGp(0.2)
    X, Y = generateData()
    x = np.linspace(-2*np.pi, 2*np.pi, 800)
    list = np.arange(1.5,2.5,0.5)
    for l in list:

        mu, sigma = computePosterior(x,X,Y,l)
        #plot observations
        plt.plot(X, Y,'ro')
        plt.plot(x,np.sin(x), color = 'green')
        mu = np.reshape(mu, (800,))
        plt.plot(x,mu, color = 'blue')
        upper = mu + 2*np.sqrt(sigma.diagonal())
        lower = mu - 2*np.sqrt(sigma.diagonal())
        ax = plt.gca()
        ax.fill_between(x, upper, lower, facecolor='pink', interpolate=True, alpha=0.1)
        #plt.title(title)
        plt.show()

plotSamplePos()










