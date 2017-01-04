############################################################# 
## Stat 202A - Homework 4
## Author: 
## Date : 
## Description: This script implements stagewise regression
## (epsilon boosting)
#############################################################

#############################################################
## INSTRUCTIONS: Please fill in the missing lines of code
## only where specified. Do not change function names, 
## function inputs or outputs. You can add examples at the
## end of the script (in the "Optional examples" section) to 
## double-check your work, but MAKE SURE TO COMMENT OUT ALL 
## OF YOUR EXAMPLES BEFORE SUBMITTING.
##
## Very important: Do not change the working directory
## in your code. If you do, I will be unable to grade your 
## work since Python will attempt to change my working directory
## to one that does not exist.
#############################################################

######################################
## Function 1: Stagewise regression ##
######################################

import numpy as np
import matplotlib.pyplot as plt

def swRegression(X, Y, numIter = 3000, epsilon = 0.000001):
  
  # Perform stagewise regression (epsilon boosting) of Y on X
  # 
  # X: Matrix (np.array) of explanatory variables.
  # Y: Response vector (np.array)
  # numIter: Number of iterations ("T" in class notes)
  # epsilon: Update step size (should be small)
  #
  # Returns a matrix (np.array) containing the stepwise 
  # solution vector for each iteration
  
  #######################
  ## FILL IN WITH CODE ##
  #######################
  
  p=X.shape[1]
  beta_all = np.zeros((p,numIter))
  db = np.zeros(p)
  beta = np.zeros(p)
  R = Y
  for t in range(0,numIter):
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

  ## Function should output the matrix (np.array) beta_all, the 
  ## solution to the stagewise regression problem.
  ## beta_all is p x numIter
  return beta_all
  
n = 100
p = 500
s = 10
X = np.random.standard_normal((n,p))
beta_true = np.zeros(p)
beta_true[0:s] = range(1, s+1)
Y = np.add(np.dot(X,beta_true),np.random.standard_normal(n,))
swRegression(X,Y)