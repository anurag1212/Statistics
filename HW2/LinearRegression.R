Ridge <- function(X, Y, lambda)
{
  n = dim(X)[1]
  p = dim(X)[2]
  Z = cbind(rep(1,n), X, Y)
  A = t(Z)%*%Z
  D = diag(rep(lambda, p+2))
  D[p+2, p+2] = 0
  D[1, 1] = 0
  A = A + D
  S = sweep(A, p+1)
  beta =S[1:(p+1), p+2]
  return(beta)
}
