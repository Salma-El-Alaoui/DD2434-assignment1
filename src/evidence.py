__author__ = 'Salma'

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.stats import multivariate_normal

def generateDataset():
    combinations = list(product([-1, 1], repeat=9))
    sets = []
    for l in combinations:
        arr = np.asarray(l)
        grid = np.reshape(arr,(3,3))
        sets.append(grid)
    return sets

def drawDataset(dataset):
    for i in range(3):
        print("|",end="")
        for j in range(3):
            print(dataset[i][j],"|",end="")
        print()

def model(number, dataset, theta):

    if number == 0:
        return 1/512
    p=1
    for i in range(3):
        for j in range(3):
            if number == 1:
                p = p * 1/(1+np.exp(-dataset[i,j]*theta[0]*(i-1)))
            if number == 2:
                p = p * 1/(1+np.exp(-dataset[i,j]*(theta[0]*(i-1) + theta[1]*(j-1))))
            if number == 3:
                p = p * 1/(1+np.exp(-dataset[i,j]*(theta[0]*(i-1) + theta[1]*(j-1)+theta[2])))
    return p

def priorSample(modelNumber, samples):
    sigma = 1000
    cov = sigma*np.eye(modelNumber)
    mean = np.zeros(modelNumber)
    theta = np.random.multivariate_normal(mean, cov, samples)
    return theta

def computeEvidence(dataset, modelNumber, samples):
    p=0
    for i in range(len(samples)):
        p = p + model(modelNumber, dataset, samples[i])
    return p/len(samples)



def create_index_set(evidence):
    dist = np.zeros([evidence.shape[0],evidence.shape[0]])
    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            dist[i,j] = evidence[i]-evidence[j]
            if i==j:
                dist[i,j] = pow(10,4)

    L = [];
    D = np.arange(evidence.shape[0]).tolist()
    ind = evidence.argmin()
    L.append(ind)
    D.remove(ind)
    while D:
        N = []
        for i in range(len(D)):
            ind = dist[D[i],D].argmin()
            if D[ind]==L[-1]:
                N.append(D[ind])
        if not N:
            L.append(D[dist[L[-1],D].argmin()])
        else:
            L.append(N[dist[L[-1],N].argmin()])
        D.remove(L[-1])
    return L


#drawDataset(l[0])
samples1 = priorSample(1,10**4)
samples2 = priorSample(2,10**4)
samples3 = priorSample(3,10**4)
l = generateDataset()

evidence = np.zeros([4,512])

for i in range(4):
    for j in range(512):
        if i == 0:
            evidence[i][j]=computeEvidence(l[j],i,samples1)
        if i == 1:
            evidence[i][j]=computeEvidence(l[j],i,samples1)
        if i == 2:
            evidence[i][j]=computeEvidence(l[j],i,samples2)
        if i == 3:
            evidence[i][j]=computeEvidence(l[j],i,samples3)


max = np.argmax(evidence,axis=1)
min = np.argmin(evidence,axis=1)
sum = np.sum(evidence, axis=1)

index = create_index_set(np.sum(evidence,axis=0))
plt.plot(evidence[0,index],'m', label = "P($\mathcal{D}$ | ${M}_0$)")
plt.plot(evidence[1,index],'b', label= "P($\mathcal{D}$ | ${M}_1$)")
plt.plot(evidence[2,index],'r', label= "P($\mathcal{D}$ | ${M}_2$)")
plt.plot(evidence[3,index],'g', label= "P($\mathcal{D}$ | ${M}_3$)")
plt.legend()
plt.show()

