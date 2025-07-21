import numpy as np

def bias_k(X,Y):
    if Y is None:
        Y= X
    return np.ones(X.shape[0], Y.shape[0])

def linear_kernel(X,Y):
    if Y is None:
        Y = X
    return X@Y.T

def affine_kernel(X,Y=None, gamma1=1., gamma2=1.):
    if Y is None: 
        Y = X
    return gamma1*(X@Y.T) + gamma2

