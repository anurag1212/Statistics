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





n = 1000
p = 5
d = 2
sigma = 1.
IT = 1000
W_true = matrix(rnorm(d*p), nrow=p)
Z_true = matrix(rnorm(n*d), nrow=d)
epsilon = matrix(rnorm(p*n)*sigma, nrow=p)
X = W_true%*%Z_true + epsilon
sq = 1.;
XX = X%*%t(X)
W = matrix(rnorm(p*d)*.1, nrow=p)
for (it in 1:IT)
{
A = rbind(cbind(t(W)%*%W/sq+diag(d), t(W)/sq), cbind(W/sq, diag(p)))
AS = mySweep(A, d)
alpha = AS[1:d, (d+1):(d+p)]
D = -AS[1:d, 1:d]
Zh = alpha %*% X
ZZ = Zh %*% t(Zh) + D*n
B = rbind(cbind(ZZ, Zh%*%t(X)), cbind(X%*%t(Zh), XX))
BS = mySweep(B, d)
W = t(BS[1:d, (d+1):(d+p)])
#sq = mean(diag(BS[(d+1):(d+p), (d+1):(d+p)]))/n;
#sq1 = mean((X-W%*%Zh)^2)
#print(cbind(sq, sq1))
}
print("W:")
print(W)


n = 200
p = 100
d = 3
sigma = .1
prob = .2
IT = 100
lambda = .1
W_true = matrix(rnorm(p*d), nrow = p)
Z_true = matrix(rnorm(n*d), nrow = d)
epsilon = matrix(rnorm(p*n)*sigma, nrow=p)
X = W_true%*%Z_true + epsilon
R = matrix(runif(p*n)<prob, nrow = p)
W = matrix(rnorm(p*d)*.1, nrow = p)
Z = matrix(rnorm(n*d)*.1, nrow = d)
for (it in 1:IT)
{
for (i in 1:n)
{
WW = t(W)%*%diag(R[,i])%*%W+lambda*diag(d)
WX = t(W)%*%diag(R[,i])%*%X[,i]
A = rbind(cbind(WW, WX), cbind(t(WX), 0))
AS = mySweep(A, d)
Z[,i] = AS[1:d, d+1]
}
for (j in 1:p)
{
ZZ = Z%*%diag(R[j, ])%*%t(Z)+lambda*diag(d)
ZX = Z%*%diag(R[j,])%*%X[j,]
B = rbind(cbind(ZZ, ZX), cbind(t(ZX), 0))
BS = mySweep(B, d)
W[j,] = BS[1:d, d+1]
}
sd1 = sqrt(sum(R*(X-W%*%Z)^2)/sum(R))
sd0 = sqrt(sum((1.-R)*(X-W%*%Z)^2)/sum(1.-R))
#print(cbind(sd1, sd0))
}
print("Z:")
print(Z)
print("W:")
print(W)
