import numpy as np
import matplotlib.pyplot as plt

n = 50
p = 50
s = 10
T = 3000
epsilon = 0.0001
beta_all = np.zeros((p,T))
X = np.random.standard_normal((n,p))
beta_true = np.zeros(p)
beta_true[0:s] = range(1, s+1)
Y = np.add(np.dot(X,beta_true),np.random.standard_normal(n,))
db = np.zeros(p)
beta = np.zeros(p)
R = Y
for t in range(0,T):
    for j in range(0,p):
        db[j] = np.sum(R*X[:, j])
    j = np.argmax(db)
    beta[j] = beta[j] + db[j]*epsilon
    R = R - X[:,j]*db[j]*epsilon
    for i in range(0,p):
        beta_all[i][t] = beta[i]

u = np.transpose(np.dot(np.ones((1,p)),abs(beta_all)))
v = np.transpose(beta_all)
plt.figure()
plt.plot(u, v,  label='Stagewise Regression')