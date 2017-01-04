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
## Very important: Do not use the function "setwd" anywhere
## in your code. If you do, I will be unable to grade your 
## work since R will attempt to change my working directory
## to one that does not exist.
#############################################################



#####################################
## Function 1: Lasso solution path ##
#####################################

# Find the lasso solution path for various values of 
# the regularization parameter lambda.
# 
# X: Matrix of explanatory variables.
# Y: Response vector
# lambda_all: Vector of regularization parameters. Make sure 
# to sort lambda_all in decreasing order for efficiency.
#
# Returns a matrix containing the lasso solution vector 
# beta for each regularization parameter.
#n = 50
#p = 200

myLasso <- function(X, Y, lambda_all){
  
  p = dim(X)[2]
  T = 10
  L = length(lambda_all)
  beta = matrix(rep(0, p), nrow = p)
  beta_all = matrix(rep(0, p*L), nrow = p)
  sort(Y, decreasing=TRUE)
  R = Y
  ss = rep(0, p)
  for (j in 1:p)
    ss[j] = sum(X[, j]^2)
  err = rep(0, L)
  for (l in 1:L)
  {
    lambda = lambda_all[l] 
    for (t in 1:T)
    {
      for (j in 1:p)
      {
        db = sum(R*X[, j])/ss[j]
        b = beta[j]+db
        b = sign(b)*max(0, abs(b)-lambda/ss[j])
        db = b - beta[j]
        R = R - X[, j]*db
        beta[j] = b
      }
    }
    beta_all[, l] = beta
  }
  
#  matplot(t(matrix(rep(1, p), nrow = 1)%*%abs(beta_all)), t(beta_all), xlab = "Lasso Solution Paths", ylab = "Beta", type = 'l')
  
  ## Function should output the matrix beta_all, the 
  ## solution to the lasso regression problem for all
  ## the regularization parameters. 
  ## beta_all is p x length(lambda_all)
  
  return(beta_all)
}

#n=50
#p=200
#s=10
#beta_true = matrix(rep(0, p), nrow = p)
#beta_true[1:s] = 1:s
#X = matrix(rnorm(n*p), nrow=n)
#Y = X %*% beta_true + rnorm(n)
#lambda_all = (100:1)*10
#myLasso(X,Y,lambda_all)