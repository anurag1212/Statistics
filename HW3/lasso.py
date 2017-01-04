############################################################# 
## Stat 202A - Homework 3
## Author: Anurag Pande
## Date : 20/10/2016
## Description: This script implements the lasso
#############################################################

#############################################################
## INSTRUCTIONS: Please fill in the missing lines of code
## only where specified. Do not change function names, 
## function inputs or outputs. You can add examples at the
## end of the script (in the "Optional examples" section) to 
## double-check your work, but MAKE SURE TO COMMENT OUT ALL 
## OF YOUR EXAMPLES BEFORE SUBMITTING.
##
## Very important: Do not change the working directory anywhere
## in your code. If you do, I will be unable to grade your 
## work since Python will attempt to change my working directory
## to one that does not exist.
#############################################################

import numpy as np
import matplotlib.pyplot as plt
#####################################
## Function 1: Lasso solution path ##
#####################################
#n = 50
#p = 200
def myLasso(X, Y, lambda_all):
  # Find the lasso solution path for various values of 
  # the regularization parameter lambda.
  # 
  # X: Array of explanatory variables.
  # Y: Response array
  # lambda_all: Array of regularization parameters. Make sure 
  # to sort lambda_all in decreasing order for efficiency.
  #
  # Returns an array containing the lasso solution  
  # beta for each regularization parameter.
#  n = X.shape[0]  
  p = X.shape[1]
#  s = 10
  T = 10  
  L = lambda_all.shape[0]
#  beta_true = np.zeros(p)
#  beta_true[0:s] = range(1, s+1)
  lambda_all = np.sort(lambda_all)[::-1]
  beta = np.zeros(p)
  beta_all = np.zeros((p,L))
  R = Y
  ss = np.zeros(p)
  for j in range(0,p):
      ss[j]=sum(np.square(X[:,j]))

  for l in range(0,L):
      lambda_dude = lambda_all[l]
      for t in range(0,T):
          for j in range(0,p):
              db = np.sum(R*X[:, j])/ss[j]
              b = beta[j]+db
              temp = abs(b)-lambda_dude/ss[j]
              sign = np.sign(b)
              b = sign*max(0,temp)
              db = b - beta[j]
              R = R - X[:, j]*db
              beta[j] = b
      for i in range(0,200):
          beta_all[i][l] = beta[i]

  #u = np.transpose(np.dot(np.ones((1,p)),abs(beta_all)))
  #v = np.transpose(beta_all)
  #plt.figure()
  #plt.plot(u, v,  label='Spline Regression')
  
  ## Function should output the array beta_all, the 
  ## solution to the lasso regression problem for all
  ## the regularization parameters. 
  ## beta_all is p x length(lambda_all)
  return(beta_all)
  
#n=50
#p=200
#X = np.random.standard_normal((n,p))
#beta_true = np.zeros(p)
#s=10
#beta_true[0:s] = range(1, s+1)
#Y = np.add(np.dot(X,beta_true),np.random.standard_normal(n,))
#lambda_all = np.arange(1000,0,-10)
#myLasso(X,Y,lambda_all)