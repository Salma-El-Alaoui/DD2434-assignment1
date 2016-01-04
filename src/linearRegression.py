__author__ = 'Salma'


import sys
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.stats import norm
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
import seaborn as sns
from sys import platform as _platform

def plotPrior(number):
    w0 = np.linspace(-2.0, 2.0, num=number)
    w1 = np.linspace(-2.0, 2.0, num=number)
    X, Y = np.meshgrid(w0, w1)
    N, M = len(X), len(Y)
    Z = np.zeros((N, M))
    for i,(x,y) in enumerate(product(w0,w1)):
        pos = np.hstack((x, y))
        Z[np.unravel_index(i, (N,M))] =  multivariate_normal([0, 0], [[0.3,0],[0,0.3]]).pdf(pos)
    im = plt.imshow(Z,cmap='jet',extent=(-2, 2, -2, 2))
    ax = plt.gca()
    ax.grid(False)
    plt.xlabel('w1')
    plt.ylabel('w0')
    plt.show()


def computeLikelihood(number, xi, yi, posterior = None):
    w0 = np.linspace(-2.0, 2.0, num=number)
    w1 = np.linspace(-2.0, 2.0, num=number)
    X, Y = np.meshgrid(w0, w1)
    N, M = len(X), len(Y)
    Z = np.zeros((N, M))
    for i,(x,y) in enumerate(product(w0,w1)):
        pos = np.hstack((x, y))
        if posterior is None:
            Z[np.unravel_index(i, (N,M))] = norm(x*xi + y, np.sqrt(0.3)).pdf(yi) * multivariate_normal([0, 0], [[0.3,0],[0,0.3]]).pdf(pos)
        else :
            Z[np.unravel_index(i, (N,M))] = norm(x*xi + y, np.sqrt(0.3)).pdf(yi) * posterior[i]

    Z= np.reshape(Z,10000)
    indices = np.argsort(Z)[::-1][:20]
    Wsamples =[]
    for i,(x,y) in enumerate(product(w0,w1)):
        for j, index in enumerate(indices):
            if i == index :
                    Wsamples.append((x,y))
    return Z, Wsamples

def plotSamples(W):
    x = np.arange(-1.0, 1.0, 0.1)
    #y = np.arange(-1.0, 1.0, 0.1)
    fig = plt.figure()
    #plt.ylim(-1.0, 1.0)
    ax = plt.gca()
    ax.grid(False)
    for (w0,w1) in W:
        plt.plot(x,w0*x+w1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def plotLikelihood(Z):
    Z= np.reshape(Z,(100,100))
    im = plt.imshow(Z,cmap='jet',extent=(-2, 2, -2, 2))
    plt.xlabel('w1')
    plt.ylabel('w0')
    ax = plt.gca()
    ax.grid(False)
    plt.show()

def plotPosterior():
    x, y = np.mgrid[-1:1:.01, -1:1:.01]
    pos = np.dstack((x, y))
    rv = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.contourf(x, y, rv.pdf(pos), cmap='jet')
    plt.show()

def plotNormal2D(X,fid):
    nbins = 200
    H, xedges, yedges = np.histogram2d(X[:,0],X[:,1],bins=nbins)
    H = np.rot90(H)
    H = np.flipud(H)

    Hmasked = np.ma.masked_where(H==0,H)
    fig = plt.figure(fid)
    plt.pcolormesh(xedges,yedges,Hmasked, cmap='jet')
    plt.ylim([-3,3])
    plt.xlim([-3,3])
    #plt.axis([-5,5,-5,5])
    plt.show()

def pickDataPoint(w0,w1,sigma,mu):
    #sample from unif(-1,1)
    x = round(2 * np.random.random_sample() -1 ,2)
    sigma = np.sqrt(sigma)
    epsilon = sigma * np.random.randn() + mu
    y = w0*x + w1 + epsilon
    return(x,y)

def posteriorDistribution(prevPosterior , likelihood):
    for i, y in enumerate(likelihood):
        prevPosterior[i] = y * prevPosterior[i]
    return prevPosterior

# visualise the prior over W
kPrior = [[0.3,0],[0,0.3]]
muPrior = [0, 0]
N = 1000000
#plot prior
#(prior,1)
#parameters
w0 = 1.3
w1 = 0.5
sigma = 0.3
mu = 0
#pick first point

plotPrior(100)
x, y = pickDataPoint(w0,w1,sigma,mu)
Z, W = computeLikelihood(100, x,y)
plotSamples(W)
plotLikelihood(Z)
for i in range(25):
    x1, y1 = pickDataPoint(w0,w1,sigma,mu)
    Z,W = computeLikelihood(100, x1, y1, Z)
    if i==1:
        plotSamples(W)
        plotLikelihood(Z)

plotSamples(W)
plotLikelihood(Z)



