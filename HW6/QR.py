import numpy as np
from scipy import linalg
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

n = 100
p = 5
X = np.random.random_sample((n, p))
beta = np.array(range(1, p+1))
Y = np.dot(X, beta) + np.random.standard_normal(n)
Z = np.hstack((np.ones(n).reshape((n, 1)), X, Y.reshape((n, 1))))
Q, R = qr(Z)
#print("Q:",Q)
#print("R:",R)
R1 = R[:p+1, :p+1]
Y1 = R[:p+1, p+1]
beta = linalg.solve(R1, Y1)
print(beta)