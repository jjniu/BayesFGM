library(statmod)

Bayesfglasso = function(
  data,
  p,
  lambda_shape = 1, 
  lambda_rate = 0.01, 
  nBurnin = 1000,
  nIter = 10000,
  verbose = TRUE,
  lambda
){
  #############################################################
  ####Gibbs sampler with functional graphical lasso prior #####
  #############################################################
  ###### input:
  ###data: functional data 
  ###p: number of nodes we consider
  ###M: number of truncated principal components
  ###lambda_sahpe: Hyperprior lambda shape for each block
  ###lambda_rate: Hyperprior lambda rate for each block
  ###nBurnin: number of burn-in samples
  ###nIter: number of samples
  ###verbose: show progress bar or not
  ###lambda: if give, do not update
  ###### output:
  ###Theta: samples for estimated precision matrix
  ##sample the Tau matrix
  sampleTau = function(Theta_F, lambda){
    p = ncol(Theta_F)
    Tau_sq = matrix(NA,p,p)

    Tau_sq = matrix(1/statmod::rinvgauss(length(c(Theta_F)), lambda/c(Theta_F),
                                        lambda^2), nrow= p, ncol= p, byrow= TRUE)
    return (Tau_sq)
  }

  ##calculate Forbenius norm blockwise
  F_norm = function(Theta, p,M){
    Theta_sq = Theta^2
    x = diag(p)%x%rep(1,M)#computation matrix
    Theta_F = t(x)%*%Theta_sq%*%x
    return (sqrt(Theta_F))
  }
  
  
  # Some parameters:
  N = dim(data)[1]
  M = as.integer(dim(data)[2]/p)

  # Centeralize data:
  for (i in 1:p*M){
    data[,i] = data[,i] - mean(data[,i], na.rm=T)
  }
  
  # sample covariance/correlation matrix:
  S = t(data) %*% data
  # Use glasso with some lambda to get initial values:
#   glassoRes = glasso::glasso(S, rho=0.3)

  # Starting values:
#   Theta = glassoRes$wi
  Theta = diag(p*M)
  
  #blockwise Frobunius norm for precision matrix
  Theta_F = F_norm(Theta, p, M)
  
  # Lasso parameter, if not given, update it:
  sampleLambda = missing(lambda)
  
  if (sampleLambda){
    lambda = rgamma(1, shape = lambda_shape, rate = lambda_rate)
  }
 
  # update Tau, latent variable for block
  Tau = sampleTau(Theta_F,lambda) %x% matrix(1, nrow = M, ncol = M)
  
  # initialize Results:
  if (sampleLambda){
    Results = list(
      Theta = array(dim = c(p*M, p*M, nIter)),
      Theta_F = array(dim = c(p, p, nIter)),
      lambda = numeric(nIter)
    )
  }else{
    Results = list(
      Theta = array(dim = c(p*M, p*M, nIter)),
      Theta_F = array(dim = c(p, p, nIter)),
      lambda = numeric(1)
    )
  }
  
  # Progress bar:
  if (verbose){
    pb = txtProgressBar(min = -nBurnin, max = nIter, style = 3,initial = 0)
  }
  
  # Start sampling:
  for (it in (-nBurnin):nIter){
    
    # For every node:
    for (i in 1:p){
      block_i = seq(((i-1)*M+1),((i-1)*M+M))
      
      ##create permutation matrix for exchange ith block and pth block
      m = 1:p
      m[p] = m[i]
      m[i] = p

      library(Matrix)
      m1 <- sparseMatrix(seq_along(m), m, x=1)
      mat_p = as.matrix(m1)%x%diag(M)
      
      ##exchange the ith and pth node
      Theta_ = mat_p%*%Theta%*%mat_p
      S_ = mat_p%*%S%*%mat_p
      Tau_ = mat_p%*%Tau%*%mat_p
      
      # For every principal componant
      for (j_ in 1:M){
        # the block at the end corner
        j = (p-1)*M +j_
        
        # sample for the ith block of column
        # using block Gibbs sampling from Wang (2012) 
        # Partition matrices:
        Theta11 = Theta_[-j,-j]
        Theta12 = Theta_[j,-j]
        Theta21 = Theta_[-j,j]
        Theta22 = Theta_[j,j]
        ##expensive step
        Theta11_inverse_bar = solve(Theta11)[seq(1,(p-1)*M),seq(1,(p-1)*M)]
        
        S11 = S_[-j,-j]
        S12 = S_[j,-j]
        S21 = S_[-j,j]
        S22 = S_[j,j]
        
        Tau11 = Tau_[-j,-j]
        Tau12 = Tau_[j,-j]
        Tau21 = Tau_[-j,j]
        Tau22 = Tau_[j,j]
        
        gamma = rgamma(1, shape = N/2 + 1, rate = (S22 + lambda^2) / 2)
        #expensive step
        #C = solve((S22 + lambda^2) * Theta11_inverse_bar+ solve(diag(c(Tau12[seq(1,(p-1)*M)]))))
        Ell= t(chol((S22 + lambda^2) * Theta11_inverse_bar 
                  + diag(1/c(Tau12[seq(1,(p-1)*M)]))))  # C = LL'
        
        #C = (C + t(C))/2##C must be a symmetric matrix
        
        ##most expensive step
        #beta = mvtnorm::rmvnorm(1,-C %*% S21[seq(1,(p-1)*M)], C)
        temp1= solve(Ell, -1*S21[seq(1,(p-1)*M)])
        mu= solve(t(Ell), temp1)
        vee= solve(t(Ell), rnorm(length(mu)))
        beta= cbind(mu + vee)
        
        Theta_[-j,j] = Theta_[j,-j] = c(beta, rep(0, M-1))
        #Theta_[j,j] = gamma + beta %*% Theta11_inverse_bar %*% t(beta)
        Theta_[j,j] = gamma + colSums(beta * (Theta11_inverse_bar %*% beta))
      }
      
      # update ith block of column/ exchange the ith and pth node
      Theta = mat_p%*%Theta_%*%mat_p
      # S = mat_p%*%S_%*%mat_p
      # Tau = mat_p%*%Tau_%*%mat_p
    }
    
    # Update tau:
    Tau_sq = sampleTau(Theta_F,lambda)
    Tau = Tau_sq %x% matrix(1, nrow = M, ncol = M)
    
    # Update lambda:
    if (sampleLambda){
      lambda_sq = rgamma(1,shape = lambda_shape + p*M + (M^2+1)*p*(p-1)/4,
                         rate = lambda_rate+ sum(diag(Theta))/2 + sum(Tau_sq[row(Tau_sq)<col(Tau_sq)]))
      lambda = sqrt(lambda_sq)
    }
    
    Theta_F = F_norm(Theta, p, M)
    
    # Store:
    if (it > 0){
      Results$Theta[,,it] = Theta
      Results$Theta_F[,,it] = Theta_F
      if(sampleLambda){
        Results$lambda[it] = lambda
      }
      #Results$pcor[,,it] = as.matrix(DensParcorr:::prec2part(Theta))
    }
    
    if (verbose){
      setTxtProgressBar(pb, it)
    }
  }

  if(!sampleLambda){
    Results$lambda = lambda
  }
  if (verbose) close(pb)
  return(Results)
}


Bayesfghorse = function(
  data,
  p,
  nBurnin = 1000,
  nIter = 1000,
  verbose = TRUE
){
  #############################################################
  ####Gibbs sampler with functional horseshoe prior #####
  #############################################################
  ###### input:
  ### data: functional data 
  ### p: number of nodes we consider
  ### M: number of truncated principal components
  ### nBurnin: number of burn-in samples
  ### nIter: number of samples
  ### verbose: show progress bar or not
  ###### output:
  ### Theta: samples for estimated precision matrix
  
  sampleLambda = function(Theta_F, Nu, tau_sq, M){
    Lambda_sq = matrix(1/rgamma(p*p, shape = (M^2+1)/2, rate = c(1/Nu+Theta_F^2/(2*tau_sq))),
                  nrow = p, ncol = p, byrow = TRUE)
    # Lambda_sq = (Lambda_sq + t(Lambda_sq))/2
    return (Lambda_sq)
  }

  sampleNu = function(Lambda){
    Nu = matrix(1/rgamma(p*p, 1, c(1+1/Lambda)), nrow = p, ncol = p, byrow = TRUE)
    # Nu = (Nu + t(Nu))/2
    return(Nu)
  }

  F_norm = function(Theta, p, M){
    ##blockwise Frobenius norm cmoputation: pM*pM -> p*p
    Theta_sq = Theta^2
    x = diag(p)%x%rep(1,M)#computation matrix
    Theta_F = t(x)%*%Theta_sq%*%x
    return (sqrt(Theta_F))
  }


  # Some parameters:
  N = dim(data)[1]
  M = as.integer(dim(data)[2]/p)
  # Center data:
  for (i in 1:p*M){
    data[,i] = data[,i] - mean(data[,i],na.rm=T)
  }
  
  # sample covariance S:
  S = t(data) %*% data
  # S = cor(data)
  
  # Use glasso with some lambda to get starting values:
#   glassoRes = glasso::glasso(cov(data,use="pairwise.complete.obs"),rho=0.3)
  
  # Starting values:
  # Inverse:
#   Theta = glassoRes$wi
  # or use identity as initial
  Theta = diag(p*M)
  
  #blockwise Frobunius norm for precision matrix
  Theta_F = F_norm(Theta, p, M)
  
  tau_sq = 1 ##initial values
  zeta = 1
  Nu = matrix(1, nrow = p, ncol = p)
  
  lambda = 1### penalty parameter for diagonal entries of precision matrix, but we can ignore
  Lambda = sampleLambda(Theta_F, Nu, tau_sq, M)
  
  Nu = sampleNu(Lambda)
  
  # initialize Results:
  Results = list(
    Theta = array(dim = c(p*M, p*M, nIter))
  )
  
  # Progress bar:
  if (verbose){
    pb = txtProgressBar(min = -nBurnin, max = nIter, style = 3,initial = 0)
  }
  
  # Start sampling:
  for (it in (-nBurnin):nIter){
    
    # For every node:
    for (i in 1:p){
      
      ##create permutation matrix for exchange ith block and pth block
      m = 1:p
      m[p] = m[i]
      m[i] = p

      library(Matrix)
      m1 <- sparseMatrix(seq_along(m), m, x=1)
      mat_p = as.matrix(m1)%x%diag(M)
      
      ##exchange the ith and pth node
      Theta_ = mat_p%*%Theta%*%mat_p
      S_ = mat_p%*%S%*%mat_p
      
      Lambda_mat = Lambda %x% matrix(1, nrow = M, ncol = M)
      Lambda_ = mat_p%*%Lambda_mat%*%mat_p
      
      Nu_mat = Nu %x% matrix(1, nrow = M, ncol = M)
      Nu_ = mat_p%*%Nu_mat%*%mat_p
      
      # For every principal component
      for (j_ in 1:M){
        # the block at the end corner
        j = (p-1)*M +j_
        
        # sample for the ith block of column
        # using block Gibbs sampling from Wang (2012) 
        # Partition matrices:
        Theta11 = Theta_[-j,-j]
        Theta12 = Theta_[j,-j]
        Theta21 = Theta_[-j,j]
        Theta22 = Theta_[j,j]
        Theta11_inverse_bar = solve(Theta11)[seq(1,(p-1)*M),seq(1,(p-1)*M)]
        
        S11 = S_[-j,-j]
        S12 = S_[j,-j]
        S21 = S_[-j,j]
        S22 = S_[j,j]
        
        Lambda11 = Lambda_[-j,-j]
        Lambda12 = Lambda_[j,-j]
        Lambda21 = Lambda_[-j,j]
        Lambda22 = Lambda_[j,j]
        
        Nu11 = Nu_[-j,-j]
        Nu12 = Nu_[j,-j]
        Nu21 = Nu_[-j,j]
        Nu22 = Nu_[j,j]
        
        gamma = rgamma(1, shape = N/2 + 1, rate = (S22) / 2)
        # C = solve((S22 + lambda^2) * Theta11_inverse_bar+ diag(1/(tau_sq*c(Lambda12[seq(1,(p-1)*M)]))))
        Ell = t(chol((S22) * Theta11_inverse_bar+ diag(1/(tau_sq*c(Lambda12[seq(1,(p-1)*M)])))))
        # C = (C + t(C))/2
        temp1= solve(Ell, -1*S21[seq(1,(p-1)*M)])
        mu= solve(t(Ell), temp1)
        vee= solve(t(Ell), rnorm(length(mu)))
        beta= cbind(mu + vee)

       # beta = mvtnorm::rmvnorm(1,-C %*% S21[seq(1,(p-1)*M)], C)##C must be a symmetric matrix
        
        Theta_[-j,j] = Theta_[j,-j] = c(beta, rep(0, M-1))
        Theta_[j,j] = gamma + colSums(beta * (Theta11_inverse_bar %*% beta))
      }
      
      # update ith block of column/ exchange the ith and pth node
      Theta = mat_p%*%Theta_%*%mat_p
    }
    
    Theta_F = F_norm(Theta, p, M)
    # Update Lambda:
    Lambda = sampleLambda(Theta_F, Nu, tau_sq, M)
    
    # Update Nu
    Nu = sampleNu(Lambda)
    
    # Update tau
    Theta_F.vec <- Theta_F[lower.tri(Theta_F)]
    Lambda.sq.vec <- Lambda[lower.tri(Lambda)]
    tau_sq = 1/rgamma(1, shape = (M^2*p*(p-1)+2)/4,  rate = 1/zeta + sum(Theta_F.vec^2/(2*Lambda.sq.vec)))
    
    # Update zeta
    zeta = 1/rgamma(1,shape = 1, rate = 1+1/tau_sq)
    
    # Store:
    if (it > 0){
      Results$Theta[,,it] = Theta
    }
    
    if (verbose){
      setTxtProgressBar(pb, it)
    }
  }

  if (verbose) close(pb)
  return(Results)
}



### example
# p = 10 ## number of nodes
# y = SVD_dense_fun(10,1000, 1)$a_hat ## pca scores of data
# res = Bayesfghorse(y,p)