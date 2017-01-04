x = matrix(runif(30625),nrow=175,ncol=175)
start.time <- Sys.time()
mySweepC(x,175)
end.time <- Sys.time()
time.taken <- end.time - start.time
print(time.taken)