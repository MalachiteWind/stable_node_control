import numpy as np

gamma = 50
eps = 0.02
q1, q2 = 0.08, 0.04
b1 = 1 - eps
b2 = 1 - eps

def sigmoid(x):
    return 1 / (1 + np.exp(-gamma * x))

def c1_in(x):
    return q1 * (1 - sigmoid(x - b1))

def c2_in(y):
    return q1 * (1 - sigmoid(y - b2))

def c1_out(y):
    return q2 * (1 - sigmoid(y - b2))

def c2_out(y):
    return q2

def two_tank_system(t, x, u):
    x1, x2 = x
    p, v = u
    x1 = np.maximum(x1, 0)
    x2 = np.maximum(x2, 0)
    dx1dt = c1_in(x1)*(1 - v)*p - c1_out(x2)*np.sqrt(x1)
    dx2dt = c2_in(x2)*v*p + c1_out(x2)*np.sqrt(x1) - q2*np.sqrt(x2)
    return np.array([dx1dt, dx2dt])