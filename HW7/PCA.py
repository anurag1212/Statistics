# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 17:28:48 2016

@author: anura
"""
import numpy as np


def qr(A):
    n, m = A.shape
    R = A.copy()
    Q = np.eye(n)
    for k in range(m-1):
        x = np.zeros((n, 1))
        x[k:, 0] = R[k:, k]
        v = x
        v[k] = x[k] + np.sign(x[k,0]) * np.linalg.norm(x)
        s = np.linalg.norm(v)
        if s != 0:
            u = v / s
            R -= 2 * np.dot(u, np.dot(u.T, R))
            Q -= 2 * np.dot(u, np.dot(u.T, Q))
    Q = Q.T
    return Q, R

def eigen_qr(A):
    T = 1000
    A_copy = A.copy()
    r, c = A_copy.shape
    V = np.random.random_sample((r, r))
    for i in range(T):
        Q, _ = qr(V)
        V = np.dot(A_copy, Q)
    Q, R = qr(V)
    return R.diagonal(), Q
    
    
def mylogistic(_x, _y):
    x = _x.copy()
    y = _y.copy()
    r, c = x.shape
    beta = np.zeros((c, 1))
    epsilon = 1e-6
    while True:
        eta = np.dot(x, beta)
        pr = exp_it(eta)
        w = pr * (1 - pr)
        z = eta + (y - pr) / w
        sw = np.sqrt(w)
        mw = np.repeat(sw, c, axis=1)
        x_work = mw * x
        y_work = sw * z
        beta_new, _, _, _ = np.linalg.lstsq(x_work, y_work)
        err = np.sum(np.abs(beta_new - beta))
        beta = beta_new
        if err < epsilon:
            break
    return beta
    
def exp_it(_x):
    x = _x.copy()
    y = 1 / (1 + np.exp(-x))
    return y

if __name__ == '__main__':
    n = 1000
    p = 5
    X = np.random.normal(0, 1, (n, p))
    #beta = np.arange(p) + 1
    beta = np.ones((p, 1))
    print(beta)
    Y = np.random.uniform(0, 1, (n, 1)) < exp_it(np.dot(X, beta)).reshape((n, 1))
    logistic_beta = mylogistic(X, Y)
    print(logistic_beta)
    
   
n = 100
p = 5
X = np.random.random_sample((n, p))
A = np.dot(X.T, X)
D, V = eigen_qr(A)
print("D: ",D.round(6))
print("V: ",V.round(6))
# Compare the result with the numpy calculation
eigen_value_gt, eigen_vector_gt = np.linalg.eig(A)
print("Val: ",eigen_value_gt.round(6))
print("Vec: ",eigen_vector_gt.round(6))
