###--------------------------------------------------------------------------###
###---------- Bayesian Neural Networks for Macroeconomic Analysis -----------###
###--------------- Hauzenberger, Huber, Klieber & Marcellino ----------------###
###------------------------ Journal of Econometrics -------------------------###
###--------------------------------------------------------------------------###
###--------------------------- Replication Code -----------------------------###
###--------------------------------------------------------------------------###
rm(list=ls())
w.dir <- ""

###--------------------------------------------------------------------------###
###------------------------------- Packages ---------------------------------###
###--------------------------------------------------------------------------###
library(MASS)
library(RcppArmadillo)
library(Rcpp)
library(coda)
library(stochvol)
library(Matrix)
library(mvtnorm)
library(truncnorm)
library(zoo)
library(dplyr)

###--------------------------------------------------------------------------###
###----------------------- Auxiliary functions ------------------------------###
###--------------------------------------------------------------------------###
source(paste0(w.dir, "functions/aux_funcs_deepNN.R"))
Rcpp::sourceCpp(paste0(w.dir,"functions/get_post_k.cpp"))
Rcpp::sourceCpp(paste0(w.dir,"functions/get_post_grad_k.cpp"))


###--------------------------------------------------------------------------###
###-------------------------- MCMC functions --------------------------------###
###--------------------------------------------------------------------------###
mcmc.setup <- list(
  "nsave"   =  1000,
  "nburn"   =  1000,
  "nthin"   =  2,
  "nuts"    =  FALSE     # Adaptive HMC sampler
)


###--------------------------------------------------------------------------###
###--------------------- Application preliminaries --------------------------###
###--------------------------------------------------------------------------###
# Data preliminaries 
target <- "INDPRO_mom" # Target variable ("CPIAUCSL_SW", "INDPRO_mom", "CE16OV_mom")
info   <- "L"            # Information set
hout   <- 2000           # Hold-out period, loop over: seq(2000, 2020+11/12,1/12)
stdz   <- TRUE           # Standardize data 

# Model preliminaries 
bnn.type <- "shlwNNflex" # Choose between: "shlwNN","shlwNNflex","shlwNNrelu","deepNN","deepNNflex","deepNNrelu"
M.lbl    <- M <- 20
cons     <- FALSE
sv       <- TRUE; bsig.SV <- 0.2

# Specify no. of layers
if (grepl("deep",bnn.type)) {
  NN.layer <- 3
} else {
  NN.layer <- 1
}

# Restrict / do not restrict activation functions 
if(bnn.type %in% c("deepNNflex", "shlwNNflex")){
  act.flex  <- TRUE  
}else if(bnn.type %in% c("deepNN", "shlwNN", "deepNNrelu", "shlwNNrelu")){
  act.flex  <- FALSE     
}
Q <- NN.layer

bnn.setup <- list(
  "sv"          = sv,
  "nsave"       = mcmc.setup$nsave,
  "nburn"       = mcmc.setup$nburn,
  "nthin"       = mcmc.setup$nthin,
  "nuts"        = mcmc.setup$nuts,  # Adaptive HMC sampler
  
  "main.spec"   = bnn.type,         # Specification
  "Q"           = Q,                # No. of hidden layers 
  "M"           = M.lbl,            # No. of neurons
  "act.flex"    = act.flex,         # Restrict activation function to be global
  "bsig.SV"     = bsig.SV,          # Prior on the state innovation variance of log-volatilities
  "t0.sig"      = 3,                # Prior DoF
  "S0.sig"      = 0.3               # Prior scaling
)


###--------------------------------------------------------------------------###
###----------------------------- Data setup ---------------------------------###
###--------------------------------------------------------------------------###

source(paste0(w.dir, "functions/data_designmat.R"))

if(M.lbl == "K") M <- K else M <- as.numeric(M)
if(M > K) cat("No. of neurons is larger than the no. of covariates!")
bnn.setup$M <- M


###--------------------------------------------------------------------------###
###---------------------- MCMC preliminaries --------------------------------###
###--------------------------------------------------------------------------###
list2env(bnn.setup,globalenv())

acf_set <- act.fc.set.all[-which(names(act.fc.set.all)=="uni")] 
if (grepl("relu",main.spec)) acf_set <- acf_set[which(names(acf_set) == "relu")]

ntot <- nburn+nsave*nthin
save.set <- seq(nthin, nsave*nthin, nthin) + nburn
save.ind <- 0

###-------------------- Get key dimensions of data --------------------------###
XX <- X

Q <- Q               # Truely hidden layers 
QQ  <- Q + 1         # Hidden layers + initial layer (i.e., X)
K <- ncol(XX)        # No. of covariates
N <- nrow(XX)        # No. of observations
R <- length(acf_set) # No. of activation functions

###--------------------------------------------------------------------------###
###------------ Get design matrices, priors, and starting values ------------###
###--------------------------------------------------------------------------###
MM <- c(K, rep(M, Q))
k_draw <- k.V <- array(0,dim=c(K,M,Q))
y.hat <- array(0,dim=c(N,M,Q)); yho.hat <- array(0,dim=c(Nho,M,Q))
X.hat <- array(0,dim=c(N,K,QQ)); Xho.hat <- array(0,dim=c(Nho,K,QQ))
X.hat[,,1] <- X; Xho.hat[,,1] <- Xho

# 1st layer has dimension K x M (input x neurons of first layer)
k_draw[,,1] <- t(matrix(runif(K*M, 0, 1), M, K))

y.hat[,,1]   <- X.hat[,1:M,2]   <- X.hat[,,1]%*%as.matrix(k_draw[,,1])/K
yho.hat[,,1] <- Xho.hat[,1:M,2] <- Xho.hat[,,1]%*%as.matrix(k_draw[,,1])/K
k.V[,,1] <- matrix(1e-10, K, M)

# Following layers have dimension M x M (neurons x neurons of next layer)
if(Q > 1){
  for(nr1 in 2:Q){
    k_draw[1:M,1:M,nr1] <- t(matrix(runif(M*M, 0, 1), M, M)) # Draw initial values
    
    y.hat[,,nr1]   <- X.hat[,1:M,nr1+1]   <-  X.hat[,1:M,nr1]    # %*%k_draw[1:M,1:M,nr1]/K    
    yho.hat[,,nr1] <- Xho.hat[,1:M,nr1+1] <-  Xho.hat[,1:M,nr1]  # %*%k_draw[1:M,1:M,nr1]/K
    k.V[1:M,,nr1]  <- matrix(10, M, M)
  } 
}

XX.hat<- cbind(X,X.hat[,1:M,QQ])
KM <- K+M

# Starting values for priors on the nonlinear coefficient part (HS prior)
b_draw  <-  matrix(0,M,1) #Nonlinear part
b.v_draw <- rep(10^2,M)
acc.k <- array(0, c(1, M, Q))
post.fc <- matrix(0,  R, 1) 

lambda.beta.mat <- matrix(0.1, M, 1)
nu.beta.mat <- matrix(0.1,  M, 1)
tau.beta <- 0.1
zeta.beta <- 0.1

# HS prior    
lam.mat <- nu.mat <- tau.mat <- zeta.mat <- list()
lam.mat[[1]]  <- matrix(0.1, K, M)
nu.mat[[1]]   <- matrix(0.1,  K, M)
tau.mat[[1]]  <- matrix(0.1, M, 1)
zeta.mat[[1]] <- matrix(0.1, M, 1)
if (Q>1) {
  for (nr1 in 2:Q) {
    lam.mat[[nr1]] <- matrix(0.1, M, M)
    nu.mat[[nr1]]     <- matrix(0.1,  M, M)
    tau.mat[[nr1]]    <- matrix(0.1, M, 1)
    zeta.mat[[nr1]]   <- matrix(0.1, M, 1)
  }
}

###---- Get design matrices, priors, and starting values for linear part ----###
obs_draw <- solve(crossprod(XX.hat) + 1e-5*diag(ncol(XX.hat)))%*%crossprod(XX.hat, y)
fit_full <- fit_lin <- fit_nn <- XX.hat%*%obs_draw
fit_nn[] <- 0
ts.plot(cbind(y,fit_full), col=c(2,1), main=c("red=True, black=approx"))
g_draw <-  matrix(0,K,1) #Linear part
sigma2.ols <- crossprod(y - XX.hat%*%obs_draw)/(N-KM)
g.v <- rep(1, K)
b.v <- rep(1, M)
g.v.inv <- 1/g.v 
b.v.inv <- 1/b.v 

# HS prior on constant coefficient part
g.lam.mat <- matrix(0.1, K, 1)
g.nu.mat  <- matrix(0.1,  K, 1)
g.tau     <- 0.1
g.zeta    <- 0.1

###--- Priors and starting values for error variances: SV versus homosk. ----###
if(sv){
  sv_priors <- specify_priors(
    mu = sv_normal(mean = 0, sd = 10), # prior on unconditional mean in the state equation
    phi = sv_beta(25, 1.5), #informative prior to push the model towards a random walk in the state equation (for comparability)
    sigma2 = sv_gamma(shape = 0.5, rate = 1/(2*bsig.SV)), # Gamma prior on the state innovation variance
    nu = sv_infinity(),
    rho = sv_constant(0))
  
  # Initialization of SV processes
  svdraw <- list(mu = 0, phi = 0.99, sigma = 0.01, nu = Inf, rho = 0, beta = NA, latent0 = 0)
}else{
  t0 <- t0.sig
  S0 <- S0.sig
}

sig2_draw <- rep(1,N)*as.numeric(0.1)
ht_draw <- log(sig2_draw)

if (substr(main.spec,1,6) == "deepNN") acf_draw <- rep(1,Q) else acf_draw <- rep(1,M)
nuts.eps <- rep(0.0001,Q)
par_list <- list(list(M_adapt = 1, M_diag = NULL))[rep(1,M)]
par_list <- list(par_list)[rep(1,Q)]

###--------------------------------------------------------------------------###
###-------------------------- Storage matrices ------------------------------###
###--------------------------------------------------------------------------###
sig2_store  <- array(NA, dim = c(nsave, N))
g_store     <- array(NA, dim = c(nsave, K))
g.v_store   <- array(NA, dim = c(nsave, K, 1))

k_store     <- array(NA, c(nsave, K, M, Q))
k.V_store   <- array(NA, c(nsave, K, M, Q))
if (substr(main.spec,1,6) == "deepNN") acf_store <- array(NA,dim=c(nsave,Q)) else acf_store <- array(NA,dim=c(nsave,M))
b_store     <- array(NA, c(nsave, M))
b.v_store   <- array(NA, c(nsave, M, 1))
y.hat_store   <- array(NA, c(nsave, N, M, Q)); dimnames(y.hat_store) <- list(1:nsave, 1:N, 1:M, 1:Q)

fit_store  <- array(NA, c(nsave, N, 4)); dimnames(fit_store) <- list(1:nsave, 1:N, c("fit + shock", "fit", "fit_lin","fit_nlin"))
pred_store <- matrix(NA, nsave,Nho)

###--------------------------------------------------------------------------###
###--------------------------------------------------------------------------###
###-------------------- START: MCMC estimation loop -------------------------###
###--------------------------------------------------------------------------###
###--------------------------------------------------------------------------###
irep <- 1
start <- Sys.time()
to.plot <- 20
for (irep in seq_len(ntot)){
  ###------------------------------------------------------------------------###
  ###---------------------------- Step 1: -----------------------------------###
  ###---------- Sample gamma and beta from a multivariate Gaussian ----------###
  ###------------------------------------------------------------------------###
  norm.sig <- 1/sqrt(sig2_draw) 
  y.lin <- y*norm.sig
  x.lin <- cbind(X, X.hat[,1:M,QQ])*norm.sig
  
  gb.V_po <- try(solve(crossprod(x.lin) + diag(c(g.v.inv, b.v.inv))), silent=F) # Conditional posterior variance-covariance of beta and gamma
  if (is(gb.V_po,"try-error")) gb.V_po <- ginv(crossprod(x.lin) + diag(c(g.v.inv, b.v.inv)))
  gb.m_po <- gb.V_po%*%crossprod(x.lin, y.lin) # Conditional posterior mean of beta and gamma
  gb_draw <- try(gb.m_po + t(chol(gb.V_po))%*%rnorm(K+M), silent=F) # Simulate from multivariate normal distribution
  if (is(gb_draw, "try-error")) gb_draw <- matrix(as.numeric(mvtnorm::rmvnorm(1, gb.m_po, as.matrix(forceSymmetric(gb.V_po)))), K+M,1)
  
  g_draw <- gb_draw[1:K,,drop = F]
  b_draw <- gb_draw[(K+1):(K+M),, drop = F]
  
  # Fit of linear part:
  fit_lin <- X%*%g_draw 
  y.nolin <- y - fit_lin
  
  
  ###------------------------------------------------------------------------###
  ###----------------------------- Step 2: ----------------------------------###
  ###------------------ Sample prior variances for gamma --------------------###
  ###----------------------- Horseshoe (HS) prior ---------------------------###
  ###------------------------------------------------------------------------###
  g_hs <- get.hs(g_draw,lambda.hs = g.lam.mat, nu.hs = g.nu.mat, tau.hs = g.tau,zeta.hs=g.zeta)
  g.lam.mat <- g_hs$lambda  # Local scales
  g.nu.mat  <- g_hs$nu
  g.tau     <- g_hs$tau     # Global scales
  g.zeta    <- g_hs$zeta
  g.v       <- g_hs$psi     # Diagonal elements of prior variance matrix
  g.v[g.v < 1e-15] <- 1e-15 # Offsetting: lower-bound
  g.v.inv <- 1/g.v          # Diagonal elements prior precision matrix
  
  
  ###------------------------------------------------------------------------###
  ###---------------------------- Step 3: -----------------------------------###
  ###------------------ Sample HS prior variances for beta ------------------###
  ###------------------------------------------------------------------------###
  hs.beta <- get.hs(b_draw,lambda.hs = lambda.beta.mat, nu.hs = nu.beta.mat, tau.hs = tau.beta,zeta.hs=zeta.beta)
  lambda.beta.mat <- hs.beta$lambda # Local scales
  nu.beta.mat <- hs.beta$nu
  tau.beta <- hs.beta$tau           # Global scales
  zeta.beta <- hs.beta$zeta
  b.v <- hs.beta$psi                # Diagonal elements of prior variance matrix
  
  b.v[b.v > 1]     <- 1             # Offsetting: upper-bound
  b.v[b.v < 1e-10] <- 1e-10         # Offsetting: lower-bound
  b.v.inv <- 1/b.v                  # Diagonal elements prior precision matrix
  
  v.obs <- c(g.v, b.v)
  v.obs.inv <- c(g.v.inv, b.v.inv)
  
  ###------------------------------------------------------------------------###
  ###----------------------------- Step 4: ----------------------------------###
  ###----------- Sample the kappa (a K x M matrix) column-wise  -------------###
  ###------------------------------------------------------------------------###
  for (nr1 in seq_len(Q)){
    for (nr2 in seq_len(M)){
      theta <- k_draw[1:MM[nr1],nr2,nr1]
      X.hat.nr <- X.hat
      wonr.slct <- setdiff(1:M, nr2)
      X.hat.wonr <- X.hat[,wonr.slct,,drop=F]
      b_draw.nr <- b_draw[nr2,,drop=F]
      b_draw.wonr <- b_draw[wonr.slct,,drop=F]
      k_draw.nr <- k_draw[,nr2,,drop=F]
      k_draw.nr[1:MM[nr1],,nr1] <- theta
      if (substr(main.spec,1,6) == "shlwNN") acf_draw.nr <- acf_draw[nr2] else acf_draw.nr <- acf_draw
      if(nuts & irep > nburn*0.2){
        nuts_draw <- NUTS_one_step(theta         = theta,
                                   iter          = irep,
                                   iterMax       = nburn*0.2+1,
                                   f             = get_post_k, #get_post_k for C++, get.post_k for R
                                   grad_f        = get_post_grad_k, #get_post_grad_k for C++, get.post.grad_k for R
                                   f_list        = list(k_draw.nr=k_draw.nr,y=y.nolin,X=XX,X.hat.nr=X.hat.nr,X.hat.wonr=X.hat.wonr,k.V=k.V[1:MM[nr1],nr2,nr1],nr1=nr1,nr2=nr2,QQ=QQ, Q=Q,MM=MM,acf_draw=acf_draw.nr,b_draw.nr=b_draw.nr,b_draw.wonr=b_draw.wonr,sig2_draw=sig2_draw,acf_set=acf_set),
                                   par_list      = par_list[[nr1]][[nr2]],
                                   delta         = 0.5, 
                                   max_treedepth = 10, 
                                   eps           = nuts.eps[nr1],
                                   verbose       = TRUE
        )
        
        k_star <- nuts_draw$theta
        par_list[[nr1]][[nr2]] <- nuts_draw$pars
        
      }else{
        k_star <- hmc_deep(theta   = theta,
                           f       = get_post_k,        #get_post_k for C++, get.post_k for R
                           grad_f  = get_post_grad_k,   #get_post_grad_k for C++, get.post.grad_k for R
                           f_list  = list(k_draw.nr=k_draw.nr,y=y.nolin,X=XX,X.hat.nr=X.hat.nr,X.hat.wonr=X.hat.wonr,k.V=k.V[1:MM[nr1],nr2,nr1],nr1=nr1,nr2=nr2,QQ=QQ, Q=Q,MM=MM,acf_draw=acf_draw.nr,b_draw.nr=b_draw.nr,b_draw.wonr=b_draw.wonr,sig2_draw=sig2_draw,acf_set=acf_set),
                           epsilon = nuts.eps[nr1], #0.1,
                           L = 20)
      }
      accept <- !identical(as.vector(k_star),as.vector(k_draw[,nr2,nr1]))
      
      if(accept){
        k_draw[1:MM[nr1],nr2,nr1] <- k_star # Update kappa
        y.hat[,nr2,nr1] <- X.hat[,1:MM[nr1],nr1]%*%k_star    # Update initial layer fit
        yho.hat[,nr2,nr1] <- Xho.hat[,1:MM[nr1],nr1]%*%k_star 
        acc.k[,nr2,nr1] <- acc.k[,nr2,nr1]+1
      }
      if (irep < 0.75*nburn){
        if (acc.k[,nr2,nr1]/irep > 0.6) nuts.eps[nr1] <- 0.99*nuts.eps[nr1]
        if (acc.k[,nr2,nr1]/irep < 0.3) nuts.eps[nr1] <- 1.01 * nuts.eps[nr1]
      }
    } # -end loop over neurons (M) for estimation of kappa
    
    ###----------------------------------------------------------------------###
    ###---------------------------- Step 5: ---------------------------------###
    ###---- Sample prior variances for elements in kappa (K x M matrix) -----###
    ###--------------- use column-wise horseshoe (HS) prior -----------------###
    ###----------------------------------------------------------------------###
    for (j in 1:M){
      k_hs.j <- get.hs(bdraw     = k_draw[1:MM[nr1],j,nr1],
                       lambda.hs = lam.mat[[nr1]][,j], 
                       nu.hs     = nu.mat[[nr1]][,j], 
                       tau.hs    = tau.mat[[nr1]][j,1],
                       zeta.hs   = zeta.mat[[nr1]][j,1])
      
      lam.mat[[nr1]][,j]       <- k_hs.j$lambda # Local scales 
      nu.mat[[nr1]][, j]       <- k_hs.j$nu
      tau.mat[[nr1]][j,1]      <- k_hs.j$tau    # Column-wise global scales 
      zeta.mat[[nr1]][j,1]     <- k_hs.j$zeta
      k.V[1:MM[nr1],j,nr1]     <- k_hs.j$psi    # Prior variances (full K x M matrix)
    } # -end loop over neurons (M) for HS prior
    
    k.V[,,nr1][k.V[,,nr1] > 10] <- 10
    k.V[,,nr1][k.V[,,nr1] < 1e-10] <- 1e-10
    
  } # -end loop over layers (Q)
  
  
  ###------------------------------------------------------------------------###
  ###----------------------------- Step 6: ----------------------------------###
  ### Draw the activation function indicator from a multinomial distribution ###
  ###------------ for a grid of possible functions (for each layer) ---------###
  ###------------------------------------------------------------------------###
  if (act.flex & substr(main.spec,1,6) == "shlwNN") {
    for (nr2 in seq_len(M)) {
      for (rr in seq_len(R)){
        X.hat[ , nr2, QQ] <- acf_set[[rr]][["func"]](y.hat[,nr2,1])
        fit.nr <- as.matrix(X.hat[,1:MM[QQ],QQ])%*%b_draw
        lik <- sum(dnorm(y.nolin, fit.nr, sqrt(sig2_draw), log = TRUE)) # Evaluate log-likelihood
        prior <- log(1/R) # Evaluate log-prior
        post.fc[rr] <- lik + prior # Get posterior likelihood
      }
      probs <- exp(post.fc -max(post.fc))/sum(exp(post.fc -max(post.fc))) # Get posterior weights 
      fc.slct <- sample(1:R, 1, prob = as.numeric(probs)) # Sample indicators from multinomial distribution
      
      X.hat[,nr2,QQ] <- acf_set[[fc.slct]][["func"]](y.hat[,nr2,1]) # Get non-linear factors X.hat
      Xho.hat[,nr2,QQ] <- acf_set[[fc.slct]][["func"]](yho.hat[,nr2,1]) # Get non-linear factors Xho.hat
      acf_draw[nr2] <- fc.slct
    } #- end loop over neurons (M) for shallow case
  } else if (act.flex & substr(main.spec,1,6) == "deepNN") {
    for (nr1 in seq_len(Q)) {
      for (rr in seq_len(R)){
        acf_draw[nr1] <- rr
        for (nn in 1:Q) X.hat[,1:MM[nn+1],nn+1] <- acf_set[[acf_draw[nn]]][["func"]](X.hat[,1:MM[nn],nn]%*%as.matrix(k_draw[1:MM[nn],,nn])) # MM[nr1]?
        
        fit.nr <- as.matrix(X.hat[,1:M,QQ])%*%b_draw
        
        lik <- sum(dnorm(y.nolin, fit.nr, sqrt(sig2_draw), log = TRUE)) # Evaluate log-likelihood
        prior <- log(1/R) # Evaluate log-prior
        post.fc[rr] <- lik + prior # Get posterior likelihood
      } # -end loop over activation functions (R)
      
      probs <- exp(post.fc -max(post.fc))/sum(exp(post.fc -max(post.fc))) # Get posterior weights 
      fc.slct <- sample(1:R, 1, prob = as.numeric(probs)) # Sample indicators from multinomial distribution
      
      X.hat[,1:MM[nr1+1],nr1+1] <- acf_set[[fc.slct]][["func"]](y.hat[,,nr1]) # Get non-linear factors X.hat
      Xho.hat[,1:MM[nr1+1],nr1+1] <- acf_set[[fc.slct]][["func"]](yho.hat[,,nr1]) # Get non-linear factors X.hat
      acf_draw[nr1] <- fc.slct
      
    } #- end loop over layers (Q) for deep case
  } else {
    for (rr in seq_len(R)){
      for (nn in 1:Q) X.hat[,1:MM[nn+1],nn+1] <- acf_set[[rr]][["func"]](X.hat[,1:MM[nn],nn]%*%as.matrix(k_draw[1:MM[nn],,nn]))
      
      fit.nr <- as.matrix(X.hat[,1:M,QQ])%*%b_draw
      
      lik <- sum(dnorm(y.nolin, fit.nr, sqrt(sig2_draw), log = TRUE)) # Evaluate log-likelihood
      prior <- log(1/R) # Evaluate log-prior
      post.fc[rr] <- lik + prior # Get posterior likelihood
    } # -end loop over activation functions (R)
    
    probs <- exp(post.fc -max(post.fc))/sum(exp(post.fc -max(post.fc))) # Get posterior weights 
    fc.slct <- sample(1:R, 1, prob = as.numeric(probs)) # Sample indicators from multinomial distribution
    
    for (nn in 1:Q) X.hat[,1:MM[nn+1],nn+1] <- acf_set[[fc.slct]][["func"]](y.hat[,,nn]) # Get non-linear factors
    for (nn in 1:Q) Xho.hat[,1:MM[nn+1],nn+1] <- acf_set[[fc.slct]][["func"]](yho.hat[,,nn]) # Get non-linear factors X.hatX.hat
    if (substr(main.spec,1,6) == "deepNN") acf_draw <- rep(fc.slct,Q) else acf_draw <- rep(fc.slct,M)
  }
  
  # Fit of neural network
  fit_nn   <- X.hat[,1:M,QQ]%*%b_draw
  fit_nn <- fit_nn - mean(fit_nn)
  
  
  ###--------------------------------------------------------------------------###
  ###------------------------------ Step 7: -----------------------------------###
  ###-------------------- Sample the error variances --------------------------###
  ###--------------------------------------------------------------------------###
  # Joint fit
  fit_full  <- fit_lin + fit_nn
  #Get residuals  
  eps <-  y - fit_full    
  
  ###------- Option 1: Sample heteroscedastic error variances (stochastic volatilities) using the "stochvol" package
  if(sv){
    svdraw <- svsample_fast_cpp(eps, startpara = svdraw, startlatent = ht_draw, priorspec = sv_priors)
    svdraw[c("mu", "phi", "sigma", "nu", "rho")] <- as.list(svdraw$para[, c("mu", "phi", "sigma", "nu", "rho")])
    ht_draw   <- t(svdraw$latent) # Extract log-variances 
    ht_draw[ht_draw < -10] <- -10
    sig2_draw <- exp(as.numeric(ht_draw)) # Get variances 
  }else{
    ####------- Option 2: Sample homoscedastic error variances from a conditionally conjugate Inverse Gamma 
    t1 <- t0 + N/2 # Posterior degree of freedoms
    S1 <- S0 + as.numeric(crossprod(eps))/2 # Posterior scaling
    sig2_cons <- 1/rgamma(1, t1, S1) # Get variance
    sig2_draw <- rep(sig2_cons, N) 
    ht_draw <- log(sig2_draw) # Define log-variance
  }
  
  sig2_draw[sig2_draw > 20*sd(y)] <- 20*sd(y) # Offsetting: upper-bound
  
  ###--------------------------------------------------------------------------###
  ###------------------------------ Step 8: -----------------------------------###
  ###------------------------------ Storage -----------------------------------###
  ###--------------------------------------------------------------------------###
  if(irep %in% save.set){
    
    save.ind <- save.ind + 1
    ###--------------------- In-sample quantities ---------------------------###
    g_store[save.ind,]     <- g_draw
    g.v_store[save.ind,,1] <- g.v # Prior variances of gamma
    
    sig2_store[save.ind,]  <- sig2_draw # Variances 
    
    b_store[save.ind, ]    <- b_draw # Linear coefficients (beta)
    b.v_store[save.ind,,1] <- b.v    # Prior variances of beta
    
    k_store[save.ind,,,]   <- k_draw # Non-linear coefficients (kappa)
    k.V_store[save.ind,,,] <- k.V # Prior variances of kappa
    
    acf_store[save.ind,]   <- acf_draw # Activation function indicators
    
    y.hat_store[save.ind,,,] <- y.hat
    
    ###--------------------------- In-sample fit ----------------------------###
    fit_store[save.ind,,4] <- fit_nn # Neural network fit
    fit_store[save.ind,,3] <- fit_lin # Linear fit
    fit_store[save.ind,,1] <- fit_full + rnorm(N, 0, sqrt(sig2_draw)) # Full fit + shock
    fit_store[save.ind,,2] <- fit_full # Full fit
    
    ###------------------- Predictions and loss measures --------------------###
    pred_m   <- as.numeric(Xho%*%g_draw + Xho.hat[,1:M,QQ]%*%b_draw)  # Predictive mean
    if(sv){# Predictive variance
      pred_h <- svdraw$para[,"mu"] + svdraw$phi*(ht_draw[N] - svdraw$mu) + rnorm(1, 0, svdraw$sigma)
      pred_V <- exp(pred_h)
    }else{
      pred_V <- sig2_draw[N] 
    }
    pred_draw <- pred_m + rnorm(Nho,0,sqrt(pred_V)) # Draw from a Gaussian
    
    pred_store[save.ind,] <- pred_draw # Store draw from predictive density 
  
  }
  
  if(irep %% to.plot == 0){
    par(mfrow=c(1,2))
    ts.plot(sqrt(sig2_draw), main = "Sigma", ylab = "")
    ts.plot(cbind(y, fit_full, fit_lin, fit_nn), col=c(1,2,3,4), lty = c(1,1,2,2), main = "Fit", ylab = "")
    print(irep)
  } 
  
}
end <- Sys.time()
time.min <- (ts(end)-ts(start))/60  

