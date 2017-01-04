#########################################################
## Stat 202A - Homework 6
## Author: 
## Date : 
## Description: This script implements QR decomposition
## and linear regression based on QR
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

##################################
## Function 1: QR decomposition ##
##################################

myQR <- function(A){
  
  ## Perform QR factorization on the matrix A
  ## FILL IN CODE HERE ##
  
  n = dim(A)[1]
  m = dim(A)[2]
  R = A
  Q = diag(1,n)
  for (k in 1:(m-1))
  {
    x = matrix(0,n,1)
    x[k:n,1] = R[k:n,k]
    v = x
    v[k] = x[k] + sign(x[k,1])*norm(x,type='F')
    s = norm(v,type='F')
    if(s!=0)
    {
      u = v/s
      R = R - 2*u%*%(t(u)%*%R)
      Q = Q - 2*u%*%(t(u)%*%Q)
    }
  }
  ## Function should output a list with Q.transpose and R
  return(list("Q" = t(Q), "R" = R))
}

###############################################
## Function 2: Linear regression based on QR ##
###############################################

myLM <- function(X, Y){
  
  ## X is an n x p matrix of explanatory variables
  ## Y is an n dimensional vector of responses
  ## Do NOT simulate data in this function. n and p
  ## should be determined by X.
  ## Use myQR inside of this function
  
  ## FILL CODE HERE ##
  
  n = dim(X)[1]
  p = dim(X)[2]
  Z = cbind(matrix(1,n,1),X,matrix(Y,nrow=n,ncol=1))
  output = myQR(Z)
  R_o=output$R
  R1 = R_o[1:(p+1),1:(p+1)]
  Y1 = R_o[1:(p+1),(p+2)]
  beta_ls = solve(R1,Y1)
  ## Function returns beta_ls, the least squares
  ## solution vector
  return(beta_ls)
  
}

n = 100
p = 5
X = matrix(rnorm(n*p),nrow=n)
beta = matrix(c(1),p,1)
Y = X%*%beta + rnorm(n)
print(myLM(X,Y))