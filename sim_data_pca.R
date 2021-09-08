###Simulation data generation: 
##these are two ways for pca scores calculation.
## pace_sparse_fun is for sparse data generation and calculating fpca scores by pace algorithm
## svd_dense_fun is for dense data generation and calculating fpca scores by svd method


# loadRData <- function(fileName){
#   #loads an RData file, and returns it
#   load(fileName)
#   get(ls()[ls() != "fileName"])
# }

F_norm = function(Theta, p,M){
  ## calculate Forbenius norm blockwise
  Theta_sq = Theta^2
  x = diag(p)%x%rep(1,M)#computation matrix
  Theta_F = t(x)%*%Theta_sq%*%x
  return (sqrt(Theta_F))
}

Network_gen = function(p, Net.num){
    ## generate two specific underlying networks
    Network_1 = function(p){
        #model 1: block banded theta
        Net.1 = diag(p)
        Net.1[abs(row(Net.1) - col(Net.1)) == 1] = 0.4
        Net.1[abs(row(Net.1) - col(Net.1)) == 2] = 0.2

        return (Net.1)
    }

    Network_2 = function(p){
        #model 2: block banded with some isolated nodes
        #p needs to be >= 6
        Net.2 = diag(p)
      subnet = Network_1(6)

        for (i in seq(1, floor(p/6), by = 2)){
          block = seq((i-1)*6+1, (i-1)*6+6)
          Net.2[block, block] = subnet
        }
        return (Net.2)
    }

    if(Net.num == 1){

      Net = Network_1(p)
 
     }else if(Net.num == 2){

      Net = Network_2(p)

     }

    return (Net)
}

pace_sparse_fun = function(p, N, Net.num){
  ##################################################################
  ##function for dense data generation and PACE PC score computation
  ##################################################################

  n.b = 5 ##use first 5 fouries-basis
  n.dense = 100 #number of measurement on [0,1] for densen case
  n.sparse = 9 #number of measurement on [0,1] for sparse case
  noise = T #whether there are added noise

  time.0 = seq(0, 1, length.out = n.dense) ##use dense, regular as example
  
  ### generate ground true
  Net = Network_gen(p, Net.num)
  Theta.true = Net %x% diag(n.b)

  print("----- Simulation Data Generation -----")
  ######generate delta for each obeservation from zero-mean multivariate Gaussian with Theta.true
  library(MASS)
  Sigma.true = solve(Theta.true)
  # library(matrixcalc)
  # Sigma.true = (Sigma.true + t(Sigma.true))/2
  # is.positive.definite(Sigma.true)
  
  delta = mvrnorm(n = N, mu = rep(0, n.b*p),  Sigma = Sigma.true)

  ##for dense, regular sampled functional data
  library(fda)
  time = time.0

  ################ create N observations
  G = rep(-99, N*length(time.0)*p)
  dim(G) = c(N, length(time.0), p)

  #creates a fourier basis on [0 1] with 5 basis functions
  fbasis = create.fourier.basis(c(0,1),n.b)
  f.basis = eval.basis(time.0, fbasis)

  for (i in 1:N){
    for (j in 1:p){
      block = seq(((j-1)*n.b+1),((j-1)*n.b+n.b))
      G[i,,j] = f.basis%*%delta[i, block]+ ifelse(rep(noise,length(time)),rnorm(length(time), 0, 0.5),rep(0,length(time)))
    }
  }

  ##mean-zero for each node
  G_mean = matrix(-99, length(time.0), p)
  for (j in 1:p){
    G_mean[,j] = apply(G[,,j],2, mean)
  }
  print("----- Use PACE Algorithm -----")
  ###employ PACE to compute PC scores
  library(fdapace)
  ##create Y-true which are functions at each node
  a_hat2 = c()
  M2 = c()
  
  ##we can use Sparsify to create sparse, irregular functional data
  for (j in 1:p){
    yTrue = c()
    time.vec = c()
    for (i in 1:N){
      ##for different subject and node the measurements are different 
      time.sparse = sort(sample(1:length(time.0), n.sparse))
      time.vec = c(time.vec, time.0[time.sparse])
      yTrue = rbind(yTrue, G[i,time.sparse,j]-G_mean[time.sparse,j])
    }

    FPCA_input = MakeFPCAInputs(IDs = rep(1:N, each = n.sparse), 
                                tVec = time.vec, yVec = as.vector(t(yTrue)))

    FPCA_res <- FPCA(FPCA_input$Ly, FPCA_input$Lt)
    
    M_ = SelectK(FPCA_res, criterion = 'FVE', FVEthreshold = 0.95)$K
    
    M2 = c(M2,M_)##record the selected M by PACE
    a_hat2 = cbind(a_hat2, FPCA_res$xiEst[,1:M_])
  }
  M.s2 = min(M2)

  a_hat = a_hat2[, 1:M.s2]
  for (i in 2:p){
    a_hat = cbind(a_hat, a_hat2[, (1:M.s2)+ (sum(M2[1:i-1]))])
  }

  results = list()
  results$a_hat = a_hat
  results$Net = Net

  return (results)
}


SVD_dense_fun = function(p, N, Net.num){
  ##################################################################
  ##function for dense data generation and SVD PC score computation
  ##################################################################

  noise = T #whether there are added noise
  n.dense = 100 #number of measurement on [0,1] for densen case
  n.b = 5 ##use first 5 fouries-basis
  time.0 = seq(0, 1, length.out = n.dense) ##use dense, regular as example

  ### generate ground true
  Net = Network_gen(p, Net.num)
  Theta.true = Net %x% diag(n.b)

  library(matrixcalc)
  if (!is.positive.definite(Theta.true)) {
    stop("The designed precision matrix is not positive definite!")
  }

  ######generate delta for each obeservation from zero-mean multivariate Gaussian with Tehta.true
  library(MASS)
  Sigma.true = solve(Theta.true)
  delta = mvrnorm(n = N, mu = rep(0, n.b*p),  Sigma = Sigma.true)

  ##for dense, regular sampled functional data
  library(fda)
  time = time.0
  print("----- Simulation Data Generation -----")
  ################ create N observations
  G = rep(-99, N*length(time)*p)
  dim(G) = c(N, length(time), p)

  #creates a fourier basis on [0 1] with 5 basis functions
  fbasis = create.fourier.basis(c(0,1),n.b)
  f.basis = eval.basis(time, fbasis)

  for (i in 1:N){
    for (j in 1:p){
      block = seq(((j-1)*n.b+1),((j-1)*n.b+n.b))
      G[i,,j] = f.basis%*%delta[i, block]+ ifelse(rep(noise,length(time)),rnorm(length(time), 0, 0.5),rep(0,length(time)))
    }
  }
  ##plot the functional data for the first subject
  # matplot(time, G[4,,], type = "l")

  ###regular SVD for computing PCA scores
  ##mean funciton for each node
  G_mean = matrix(-99, length(time), p)
  for (j in 1:p){
    G_mean[,j] = apply(G[,,j],2, mean)
  }
  # matplot(time, G_mean, type = "l")
  print("----- Use SVD Method -----")
  #compute empirical estimator for covariance matrix
  ##eigen-decomposition for covariance matrix
  K_hat = rep(-99, length(time)*length(time)*p)
  dim(K_hat) = c(p, length(time),length(time))

  lambda = rep(-99, length(time)*p)
  dim(lambda) = c(p, length(time))

  phi = rep(-99, length(time)*length(time)*p)
  dim(phi) = c(p, length(time), length(time))

  for (j in 1:p){
    K_hat[j,,] = cov(t(t(G[,,j])-G_mean[,j]))
    lambda[j,] = eigen(K_hat[j,,])$values
    phi[j,,] = eigen(K_hat[j,,])$vectors
  }

  #percentage of variability
  P_v = rep(-99, length(time)*p)
  dim(P_v) = c(p, length(time)) 
  M = rep(-99, p)
  for(j in 1:p){
    for (t in 1: length(time)){
      ##cumulative percentage of total variance 
      P_v[j,t] = sum(lambda[j,1:t])/sum(lambda[j,])
    }
    M[j] = which(P_v[j,]>0.9)[1]##select M to cover 90% variance
  }
  M.s = min(M)
  # M.s
  ##dot product function
  dot_fun = function(a,b){
    return (t(a)%*%b)
  }

  ###compute pricipal component score for all observations
  a_hat = rep(-99, N*M.s*p)
  dim(a_hat) = c(N, M.s*p)

  for (i in 1:N){
    for (j in 1:p){
      block_j = seq((j-1)*M.s+1, (j-1)*M.s+M.s)
      a_hat[i,block_j] = apply(phi[j,,1:M.s], 2, FUN = dot_fun, b = G[i,,j]-G_mean[,j])
    }
  }

  results = list()
  results$a_hat = a_hat
  results$Net = Net

  return (results)
}
# y = pace_sparse_fun(10, 1000, 1)$a_hat
# y = SVD_dense_fun(10,1000, 1)$a_hat
# S = cov(y)
# S_inv = solve(S)
# library(ggcorrplot)
# ##true precision
# mat = abs(DensParcorr:::prec2part(S_inv))
# ggcorrplot(t(mat[nrow(mat):1,]), show.legend = FALSE) +labs(title = "True precision matrix")

