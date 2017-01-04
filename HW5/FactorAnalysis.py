############################################################# 
## Stat 202A - Homework 5
## Author: Anurag Pande - 604749647
## Date : 11/03/2016
## Description: This script implements factor analysis and 
## matrix completion
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


def mySweep(A, m):
    """
    Perform a SWEEP operation on A with the pivot element A[m,m].
    
    :param A: a square matrix.
    :param m: the pivot element is A[m, m].
    :returns a swept matrix. Original matrix is unchanged.
    """
    
    ## No need to change anything here
    B = np.copy(A)   
    n = B.shape[0]
    for k in range(m):
        for i in range(n):
            for j in range(n):
                if i!=k and j!=k:
                    B[i,j] = B[i,j] - B[i,k]*B[k,j] / B[k,k]
        for i in range(n):
            if i!=k:
                 B[i,k] = B[i,k] / B[k,k]
        for j in range(n):
            if j!=k:
                B[k,j] = B[k,j] / B[k,k]
        B[k,k] = -1/B[k,k]
    
    return(B)

    
def factorAnalysis(n = 10, p = 5, d = 2, sigma = 1, nIter = 1000):
   
    """
    Perform factor analysis on simulated data.
    Simulate data X from the factor analysis model, e.g. 
    X = Z_true * W.T + epsilon
    where W_true is p * d loading matrix (numpy array), Z_true is a n * d matrix 
    (numpy array) of latent factors (assumed normal(0, I)), and epsilon is iid 
    normal(0, sigma^2) noise. You can assume that W_true is normal(0, I)
    
    :param n: Sample size.
    :param p: Number of variables
    :param d: Number of latent factors
    :param sigma: Standard deviation of noise
    :param nIter: Number of iterations
    """

    ## FILL CODE HERE
    
    W_true = np.random.standard_normal((p,d))
    Z_true = np.random.standard_normal((d,n))
    epsilon = np.random.standard_normal((p,n))*sigma
    X = np.dot(W_true,Z_true) + epsilon
    sq = 1
    XX = np.dot(X,np.transpose(X))
    w = np.random.standard_normal((p,d))*0.1
    for it in range(0,nIter):
        
        A= np.vstack((
        np.hstack((
        np.dot(np.transpose(w),w)/sq+np.identity(d)
        ,np.transpose(w)/sq))
        ,np.hstack((w/sq, np.identity(p)))))
        
        AS = mySweep(A, d)
        alpha = AS[0:d,d:d+p]
        D = -AS[0:d,0:d]
        Zh = np.dot(alpha, X)
        ZZ = np.dot(Zh,np.transpose(Zh)) + D*n
        B = np.vstack((np.hstack((ZZ,np.dot(Zh,np.transpose(X))))
        ,np.hstack((np.dot(X,np.transpose(Zh)),XX))
        ))
        BS = mySweep(B,d)
        w = np.transpose(BS[0:d,d:d+p])
        #sq = np.mean(np.diag(BS[d:d+p,d:d+p]))/n;
        #sq1 = np.mean((X-np.dot(W,Zh))**2)
        #print(np.hstack((sq,sq1)))


    
    ## Return the p * d np.array w, the estimate of the loading matrix
    return(w)

    
def matrixCompletion(n = 200, p = 100, d = 3, sigma = 0.1, nIter = 100,
                     prob = 0.2, lam = 0.1):
   
    """
    Perform matrix completion on simulated data.
    Simulate data X from the factor analysis model, e.g. 
    X = Z_true * W.T + epsilon
    where W_true is p * d loading matrix (numpy array), Z_true is a n * d matrix 
    (numpy array) of latent factors (assumed normal(0, I)), and epsilon is iid 
    normal(0, sigma^2) noise. You can assume that W_true is normal(0, I)
    
    :param n: Sample size.
    :param p: Number of variables
    :param d: Number of latent factors
    :param sigma: Standard deviation of noise
    :param nIter: Number of iterations
    :param prob: Probability that an entry of the matrix X is not missing
    :param lam: Regularization parameter
    """

    ## FILL CODE HERE

    W_true = np.random.standard_normal((p,d))
    Z_true = np.random.standard_normal((d,n))
    epsilon = np.random.standard_normal((p,n))*sigma
    X = np.dot(W_true,Z_true) + epsilon
    R = np.random.uniform(0,1,(p,n))<prob
    W = np.random.standard_normal((p,d))*0.1
    Z = np.random.standard_normal((d,n))*0.1
    for it in range(0,nIter):
        for i in range(0,n):
            WW1 = np.dot(np.transpose(W),np.diag(R[:,i]))
            WW = np.dot(WW1,W)+lam*np.identity(d)
            WX1 = np.dot(np.transpose(W),np.diag(R[:,i]))
            WX = np.dot(WX1,X[:,i])
            t1 = np.column_stack((WW,WX))
            t2 = np.hstack((np.transpose(WX),0))
            A = np.vstack((t1,t2))
            AS = mySweep(A,d)
            Z[:,i] = AS[0:d,d]
        for j in range(0,p):
            ZZ1 = np.dot(Z,np.diag(R[j,:]))
            ZZ = np.dot(ZZ1,np.transpose(Z))+lam*np.identity(d)
            ZX1 = np.dot(Z,np.diag(R[j,:]))
            ZX = np.dot(ZX1,X[j,:])
            u1 = np.column_stack((ZZ,ZX))
            u2 = np.hstack((np.transpose(ZX),0))
            B = np.vstack((u1,u2))
            BS = mySweep(B,d)
            W[j,:] = BS[0:d,d]
        #sd1 = np.sqrt(np.sum(R*(X-np.dot(W,Z))**2)/np.sum(R))
        #sd0 = np.sqrt(np.sum((1-R)*(X-np.dot(W,Z))**2)/np.sum(1-R))
        #print(np.hstack((sd1, sd0)))
            
    ## Return estimates of Z and W (both numpy arrays)
    return Z, W  

###########################################################
### Optional examples (comment out before submitting!!) ###
###########################################################

#print("W: ",factorAnalysis())
#a,b=matrixCompletion()
#print("Z:",a)
#print("W:",b)