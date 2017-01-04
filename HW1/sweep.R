############################################################# 
## Stat 202A - Homework 1
## Author: Anurag Pande
## Date : October 1 2016
## Description: This script implements the sweep operator as
## well as Gauss-Jordan elimination in both plain and
## vectorized form
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


## ~~~~~~~~~~~~~~~~~~~~~ ##
## ~~~~~ Problem 1 ~~~~~ ##
## ~~~~~~~~~~~~~~~~~~~~~ ##  


#################################
## Function 1: Sweep operation ##
#################################

mySweep <- function(A, m){
  
  # Perform a SWEEP operation on the square matrix A with the 
  # pivot element A[m,m].
  # 
  # A: a square matrix.
  # m: the pivot element is A[m, m].
  # Returns a swept matrix.
  
  #######################################
  ## FILL IN THE BODY OF THIS FUNCTION ##
  #######################################
  
  
  n <- dim(A)[1]
  k <- m
  for(k in 1:m)
  {
    for(i in 1:n)
    {
      for(j in 1:n)
      {
        if(i != k && j != k)
        {
          A[i,j]<-A[i,j]-(A[i,k]*A[k,j])/A[k,k]
        }
      }
    }
    for(i in 1:n)
    {
      if(i != k)
      {
        A[i,k] <- A[i,k]/A[k,k]
      }
    }
    for(j in 1:n)
    {
      if(j != k)
      {
        A[k,j] <- A[k,j]/A[k,k]
      }
    }
    A[k,k] <- -1/A[k,k]
}
  
  
  ## The output is the modified matrix A
  return(A)
  
}

#########################################################
## Function 2: Use sweep operation to find determinant ##
#########################################################

myDet <- function(A){

  ## Use the sweep operator to find the determinant of 
  ## the square matrix A
  #
  # A: a square matrix.
  # Returns the determinant of A.
  
  #######################################
  ## FILL IN THE BODY OF THIS FUNCTION ##
  #######################################
  
  Det <- 1
  n <- dim(A)[1]

  for(k in 1:n)
  {
    for(i in 1:n)
    {
      for(j in 1:n)
      {
        if(i != k && j != k)
        {
          A[i,j]<-A[i,j]-(A[i,k]*A[k,j])/A[k,k]
        }
      }
    }
    for(i in 1:n)
    {
      if(i != k)
      {
        A[i,k] <- A[i,k]/A[k,k]
      }
    }
    for(j in 1:n)
    {
      if(j != k)
      {
        A[k,j] <- A[k,j]/A[k,k]
      }
    }
    Det <- Det*A[k,k]
    A[k,k] <- -1/A[k,k]
  }

  
  ## Return the determinant (a real number, aka "numeric" class)
  return(Det)
}




## ~~~~~~~~~~~~~~~~~~~~~ ##
## ~~~~~ Problem 2 ~~~~~ ##
## ~~~~~~~~~~~~~~~~~~~~~ ##  

#####################################################
## Function 3: Elementwise version of Gauss Jordan ##
#####################################################


myGaussJordan <- function(A, m){
  
  # Perform Gauss Jordan elimination on A.
  # 
  # A: a square matrix.
  # m: Number of diagonal elements to loop through.

  #######################################
  ## FILL IN THE BODY OF THIS FUNCTION ##
  #######################################

    n <- dim(A)[1]
    B <- cbind(A, diag(rep(1,n)))
    for(k in 1:m)
    {
      a = B[k,k]
      for(j in 1:(n*2))
        B[k,j] <- B[k,j]/a
      
      for(i in 1:n)
        if(i != k)
        {
          a <- B[i,k]
          for(j in 1:(n*2))
            B[i,j] <- B[i,j] - B[k,j]*a
        }
    }
  
  ## Function returns the matrix B
  return(B)
  
}

####################################################
## Function 4: Vectorized version of Gauss Jordan ##
####################################################

myGaussJordanVec <- function(A, m){
  
  # Perform Gauss Jordan elimination on A.
  # 
  # A: a square matrix.
  # m: Number of diagonal elements to loop through.

  #######################################
  ## FILL IN THE BODY OF THIS FUNCTION ##
  #######################################
  n <- dim(A)[1]
  B <- cbind(A, diag(rep(1,n)))
  for(k in 1:m)
  {
    a = B[k,k]
    B[k, ] <- B[k, ]/a
    
    for(i in 1:n)
      if(i != k)
      {
        a <- B[i,k]
        B[i, ] <- B[i, ] - B[k, ]*a
      }
  }
  
  ## Function returns the matrix B
  return(B)
  
}



########################################################
## Optional examples (comment out before submitting!) ##
########################################################

#x = matrix(runif(30625),nrow=175,ncol=175)
#start.time <- Sys.time()
#mySweep(x,175)
#end.time <- Sys.time()
#time.taken <- end.time - start.time
#print(time.taken)