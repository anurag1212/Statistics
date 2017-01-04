#########################################################
## Stat 202A - Homework 7
## Author: Anurag Pande
## Date : 17/11/2016
## Description: This script implements PCA and logistic
## regression.
#########################################################

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

## First load in the Rcpp script containing your 
## QR decomposition code. Make sure your cpp file
## is named 'QR.cpp' and is in the current 
## working directory. Finally, make sure your 
## QR function in Rcpp is named "myQRC"

## DO NOT CHANGE THE FOLLOWING 5 LINES
library(Rcpp)
filename <- 'QR.cpp'
sourceCpp(filename)
if(filename != 'QR.cpp')
  stop("I didn't bother reading the instructions!!!!!")


######################
## Function 1: PCA  ##
######################


myEigen_QR <- function(A, nIter){
  
  ## Perform PCA on matrix A using your QR function, myQRC.
  ## A: Square matrix
  ## nIter: Number of iterations
  
  ## FILL IN CODE HERE ##
  ## Don't forget to centralize A ##

  A_copy = A
  r = dim(A_copy)[1]
  c = dim(A_copy)[2]
  V = matrix(runif(r*r),nrow = r)
  for(i in 1:nIter)
  {
    output = myQRC(V)
    Q = output$Q
    V = A_copy%*%Q
  }
  output =  myQRC(V)
  V=output$Q
  R=output$R
  D = diag(R)
  ## Function should output a list with D and V, where
  ## D is a vector of eigenvalues of A, and V is the 
  ## matrix of eigenvectors of A (in the same order as 
  ## the eigenvalues in D.)
  return(list("D" = D, "V" = V))
  
}

#####################################
## Function 2: Logistic Regression ##
#####################################

## First, define expit (sigmoid) function
expit <- function(x){
  1 / (1 + exp(-x))
}

myLogistic <- function(X, Y, epsilon = 1e-6){
  
  ## X is an n x p matrix of explanatory variables
  ## Y is an n dimensional vector of binary responses
  ## (e.g. Yi is 0 or 1).
  ## Do NOT simulate data in this function. n and p
  ## should be determined by X.
  ## Use myQRC inside of this function
  
  ## FILL CODE HERE ##
  r = dim(X)[1]
  c = dim(X)[2]
  x = X
  y = Y
  
  beta_logistic = matrix(0, c, 1)
  
  while(TRUE){
    eta = x%*%beta_logistic
    pr = expit(eta)
    w = pr*(1-pr)
    z = eta+(y-pr)/w
    sw = sqrt(w)
    mw = rep(sw,c)
    x_work = mw*x
    y_work = sw*z
    z_work = cbind(matrix(1, r, 1), x_work, matrix(y_work, r, 1))
    R = myQRC(z_work)$R
    lm_output = solve(R[1:(c+1),1:(c+1)], R[1:(c+1), c+2])
    beta_new = lm_output[2:(c+1)]
    intercept = lm_output[1]
    err = sum(abs(beta_new-beta_logistic))
    beta_logistic = beta_new
    if(err < epsilon)
      break
  }
  beta_logistic = c(intercept, beta_logistic)
  ## Function returns beta_logistic, the solution to
  ## the logistic regression problem
  return(beta_logistic)
  
}

# TESTING
# 
# test_eigen <- function(){
#   n = 100
#   p = 5
#   X = matrix(runif(n*p),nrow = n)
#   n = dim(X)[1]
#   p = dim(X)[2]
#   X_t = t(X)
#   A = X_t%*%X
#   output_qr = myEigen_QR(A,1000)
#   D = output_qr$D
#   V = output_qr$V
#   print(D)
#   print(V)
#   return(NULL)
# }
# 
# test_mylog <- function(){
#   n = 1000
#   p = 5
#   X = matrix(rnorm(n*p), nrow = n)
#   beta = matrix(rep(1),nrow = p)
#   print(beta)
#   Y = matrix(runif(n),nrow=n) < expit(X%*%beta)
#   logistic_beta = myLogistic(X,Y)
#   print(logistic_beta)
# }
# 
# 
# test_eigen()
# test_mylog()