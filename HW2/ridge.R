############################################################# 
## Stat 202A - Homework 2
## Author: 
## Date : 
## Description: This script implements ridge regression as 
## well as piecewise linear spline regression.
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

#################################
## Function 1: Sweep operation ##
#################################

mySweep <- function(A, m){
  
  # Perform a SWEEP operation on A with the pivot element A[m,m].
  # 
  # A: a square matrix.
  # m: the pivot element is A[m, m].
  # Returns a swept matrix.
  
  ## Leave this function as is unless you want to make it 
  ## more efficient!
  
  n <- nrow(A)
  
  for(k in 1:m){ 
    for(i in 1:n)     
      for(j in 1:n)   
        if(i != k  & j != k)     
          A[i,j] <- A[i,j] - A[i,k]*A[k,j]/A[k,k]    
        
        for(i in 1:n) 
          if(i != k) 
            A[i,k] <- A[i,k]/A[k,k]  
          
          for(j in 1:n) 
            if(j != k) 
              A[k,j] <- A[k,j]/A[k,k]
            
            A[k,k] <- - 1/A[k,k]
  }
  
  return(A)
  
}


##################################
## Function 2: Ridge regression ##
##################################

myRidge <- function(X, Y, lambda){
  
  # Perform ridge regression of Y on X.
  # 
  # X: a matrix of explanatory variables.
  # Y: a vector of dependent variables. Y can also be a 
  # matrix, as long as the function works.
  # lambda: regularization parameter (lambda >= 0)
  # Returns beta, the ridge regression solution.

  ##################################
  ## FILL IN THIS SECTION OF CODE ##
  ##################################
  n = dim(X)[1]
  p = dim(X)[2]
  Z = cbind(rep(1,n), X, Y)
  A = t(Z)%*%Z
  D = diag(rep(lambda, p+2))
  D[p+2, p+2] = 0
  D[1, 1] = 0
  A = A + D
  S = mySweep(A, p+1)
  beta_ridge =S[1:(p+1), p+2]
  
  ## Function should output the vector beta_ridge, the 
  ## solution to the ridge regression problem. Beta_ridge
  ## should have p + 1 elements.
  return(beta_ridge)
  
}


##############################################################
## Optional function (not graded, just for testing myRidge) ##
##############################################################

testRidge <- function(){
  
  ## You can edit this function however you like (or not at all). 
  ## I just included it here to help you check if your ridge 
  ## function is working properly.
  
  ## Load up "trees" data
  my.data <- trees
  
  ## Let X be the Girth and Height of trees
  ## X needs to be a matrix
  X <- as.matrix(trees[,1:2])
  
  ## Let Y be the Volume of the tree
  Y <- trees$Volume
  
  ## Compare ridge solution when lambda = 0 to ordinary linear regression
  beta_ls    <- lm(Volume ~ Girth + Height, data = my.data)$coeff
  beta_ridge <- myRidge(X, Y, 0)
  
  if(sum(beta_ridge - beta_ls) <= 10^(-5))
    cat("Looks like myRidge is working properly when lambda = 0! \n")
  
  ## You can test around some more
  
}


mySplineTest <- function(x, Y, lambda, p = 100, beta_spline){

  n = length(x)
  X <- matrix(x, nrow=n)
  
  for ( k in (1:(p-1))/p)
    X <- cbind(X,(x>k)*(x-k))
  
  Yhat <- cbind(rep(1,n),X)%*% beta_spline

  output <- list(beta_spline = beta_spline, predicted_y = Yhat)
  return(output)
  
}


####################################################
## Function 3: Piecewise linear spline regression ##
####################################################



mySpline <- function(x, Y, lambda, p = 100){

  
  # Perform spline regression of Y on x.
  # 
  # x: An n x 1 vector or n x 1 matrix of explanatory variables.
  # You can assume that 0 <= x_i <= 1 for i=1,...,n.
  # Y: An n x 1 vector of dependent variables. Y can also be an 
  # n x 1 matrix, as long as the function works.
  # lambda: regularization parameter (lambda >= 0)
  # p: Number of cuts to make to the x-axis.
  
  
  n = length(x)
  X <- matrix(x, nrow=n)
  
  for ( k in (1:(p-1))/p)
    X <- cbind(X,(x>k)*(x-k))

  beta_spline <- myRidge(X,Y,lambda)
  Yhat <- cbind(rep(1,n),X)%*% beta_spline
  
  
  ##################################
  ## FILL IN THIS SECTION OF CODE ##
  ##################################
  
  ## Function should return a list containing two elements:
  ## The first element of the list is the spline regression
  ## beta vector, which should be p + 1 dimensional (here, 
  ## p is the number of cuts we made to the x-axis).
  ## The second element is y.hat, the predicted Y values
  ## using the spline regression beta vector. This 
  ## can be a numeric vector or matrix.

  
  output <- list(beta_spline = beta_spline, predicted_y = Yhat)
  return(output)
 
}

########################################################
## Optional examples (comment out before submitting!) ##
########################################################
n=100

sigma=0.1
lambda=1000

p=100
X <- runif(n)

train_x <- X[1:80]
test_x <- X[81:100]

train_x <- sort (train_x)
test_x <- sort (test_x)

train_y <- train_x^2 + rnorm(80) * sigma
test_y <- test_x^2 + rnorm(20) * sigma

op_train <- mySpline(train_x,train_y,lambda,p)
beta_spl <- op_train$beta_spline
training_error = sum(abs(train_y - op_train$predicted_y))/length(train_y)

op_test <- mySplineTest(test_x,test_y,lambda,p,beta_spl)
testing_error = sum(abs(test_y - op_test$predicted_y))/length(test_y)

print(training_error)
print(testing_error)

tr=c(0.07105358,0.07071239,0.07752532,0.08270616,0.1061344,0.2157368)
ts=c(0.07870062,0.0753857,0.06756782,0.103775,0.1051672,0.2620837)
lambdas=c(1,2,3,4,5,6)
plot(ts,lambdas, col="blue", main = "Testing errors vs Lambdas", xlab = "X", ylab = "Y", type = "l")
#points(X,y)
#lines(X,op$predicted_y, type="l", col="green")
#points(X,op$predicted_y, col = "green")
#legend('topleft', c("Random","Predicted"),lty=1, col=c('blue', 'green'), bty='n', cex=.75)