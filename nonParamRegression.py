__author__ = 'HP'

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
    return np.exp(-cdist(X, Y, 'sqeuclidean')/(l*l)) + 0.5*np.ones((len(X),len(X)))

def computePosterior(xStar, X, Y,l):
    Xstar = np.array([xStar])
    X = X[:,None]
    Xstar = Xstar[:,None]
    k = kernel(Xstar,X,l)
    C = np.linalg.inv(kernel(X,X,l))
    t = Y[:,None]
    mu = np.dot(np.dot(k,C),t)

    c = kernel(Xstar, Xstar,l)
    sigma = c- np.dot(np.dot(k,C),np.transpose(k))
    return mu, np.sqrt(sigma)

def plotPosterior():
    #plotGp(0.2)
    X, Y = generateData()
    x = np.linspace(-2*np.pi, 2*np.pi, 800)
    list = np.arange(1.0,3.0,0.5)
    for l in list:
        mean = []
        deviation = []
        title = 'length-scale '+str(l)
        for data in x:
            mu, sigma = computePosterior(data,X,Y,l)
            mean.append(mu[0][0])
            deviation.append(sigma[0][0])
        #plot observations
        plt.plot(X, Y,'ro')
        plt.plot(x,np.sin(x), color = 'green')
        #plot mean
        plt.plot(x,mean, color = 'blue')
        #plot uncertainty
        upper = np.asarray(mean)+ 2*np.asarray(deviation)
        lower = np.asarray(mean)- 2*np.asarray(deviation)
        ax = plt.gca()
        ax.fill_between(x, upper, lower, facecolor='pink', interpolate=True, alpha=0.1)
        #plt.title(title)
        plt.show()

plotPosterior()










