
# No-U-Turn Sampler - an R implementation (from https://github.com/kasparmartens/NUTS)
NUTS_one_step <- function(theta, iter, iterMax, f, grad_f, f_list, par_list, delta = 0.5, max_treedepth = 10, eps = 1, verbose = TRUE){
  kappa <- 0.75
  t0 <- 10
  gamma <- 0.05
  M_adapt <- par_list$M_adapt
  if(is.null(par_list$M_diag)){
    M_diag <- rep(1, length(theta))
  }else{
    M_diag <- par_list$M_diag
  }
  
  if(iter <= iterMax){
    eps <- find_reasonable_epsilon(theta, f, grad_f, f_list, M_diag, eps = eps, verbose = verbose)
    mu <- log(10*eps)
    H <- 0
    eps_bar <- 1
  }else{
    eps <- par_list$eps
    eps_bar <- par_list$eps_bar
    H <- par_list$H
    mu <- par_list$mu
  }
  
  r0 <- rnorm(length(theta), 0, sqrt(M_diag))
  u <- runif(1, 0, exp(f(theta = theta,f_list = f_list) - 0.5 * sum(r0^2/M_diag)))
  if(is.nan(u)){
    warning("NUTS: sampled slice u is NaN")
    u <- runif(1, 0, 1e5)
  }
  theta_minus <- theta
  theta_plus <- theta
  r_minus <- r0
  r_plus <- r0
  j=0
  n=1
  s=1
  if(iter > M_adapt){
    eps <- runif(1, 0.9*eps_bar, 1.1*eps_bar)
  }
  while(s == 1){
    # choose direction {-1, 1}
    direction <- sample(c(-1, 1), 1)
    if(direction == -1){
      temp <- build_tree(theta_minus, r_minus, u, direction, j, eps, theta, r0, f, grad_f, f_list, M_diag)
      theta_minus <- temp$theta_minus
      r_minus <- temp$r_minus
    } else{
      temp <- build_tree(theta_plus, r_plus, u, direction, j, eps, theta, r0, f, grad_f, f_list, M_diag)
      theta_plus <- temp$theta_plus
      r_plus <- temp$r_plus
    }
    if(is.nan(temp$s)) temp$s <- 0
    if(temp$s == 1){
      if(runif(1) < temp$n / n){
        theta <- temp$theta
      }
    }
    n <- n + temp$n
    s <- check_NUTS(temp$s, theta_plus, theta_minus, r_plus, r_minus)
    j <- j + 1
    if(j > max_treedepth){
      warning("NUTS: Reached max tree depth")
      break
    }
  }
  if(iter <= M_adapt){
    H <- (1 - 1/(iter + t0))*H + 1/(iter + t0) * (delta - temp$alpha / temp$n_alpha)
    log_eps <- mu - sqrt(iter)/gamma * H
    eps_bar <- exp(iter**(-kappa) * log_eps + (1 - iter**(-kappa)) * log(eps_bar))
    eps <- exp(log_eps)
  } else{
    eps <- eps_bar
  }
  
  return(list(theta = theta,
              pars = list(eps = eps, eps_bar = eps_bar, H = H, mu = mu, M_adapt = M_adapt, M_diag = M_diag)))
}

leapfrog_step <- function(theta, r, eps, grad_f, f_list, M_diag){
  r_tilde <- r + 0.5 * eps * grad_f(theta     = theta,
                                    f_list    = f_list)
  
  theta_tilde <- theta + eps * r_tilde / M_diag
  r_tilde <- r_tilde + 0.5 * eps * grad_f(theta  = theta_tilde,
                                          f_list = f_list)
  list(theta = theta_tilde, r = r_tilde)
}


find_reasonable_epsilon <- function(theta, f, grad_f, f_list, M_diag, eps = 1, verbose = TRUE){
  r <- rnorm(length(theta), 0, sqrt(M_diag))
  proposed <- leapfrog_step(theta, r, eps, grad_f, f_list, M_diag)
  log_ratio <- f(theta = proposed$theta, f_list = f_list) - f(theta = theta, f_list = f_list) + 0.5*(sum(r^2 / M_diag) - sum(proposed$r^2/M_diag)) 
  alpha <- ifelse(exp(log_ratio) > 0.5, 1, -1)
  if(is.nan(alpha)) alpha <- -1
  count <- 1
  while(is.nan(log_ratio) || alpha*log_ratio > (-alpha)*log(2)){
    eps <- 2^alpha*eps
    proposed <- leapfrog_step(theta, r, eps, grad_f, f_list, M_diag)
    log_ratio <- f(theta = proposed$theta, f_list = f_list) - f(theta = theta, f_list = f_list) + 0.5*(sum(r^2 / M_diag) - sum(proposed$r^2/M_diag)) 
    count <- count + 1
    if(count > 100) {
      stop("Could not find reasonable epsilon in 100 iterations!")
    }
  }
  if(verbose) message("Reasonable epsilon = ", eps, " found after ", count, " steps")
  eps
}

check_NUTS = function(s, theta_plus, theta_minus, r_plus, r_minus){
  if(is.na(s)) return(0)
  condition1 <- crossprod(theta_plus - theta_minus, r_minus) >= 0
  condition2 <- crossprod(theta_plus - theta_minus, r_plus) >= 0
  s && condition1 && condition2
}

build_tree = function(theta, r, u, v, j, eps, theta0, r0, f, grad_f, f_list, M_diag, Delta_max = 1000){
  if(j == 0){
    proposed <- leapfrog_step(theta, r, v*eps, grad_f, f_list, M_diag)
    theta <- proposed$theta
    r <- proposed$r
    log_prob  <- f(theta = theta, f_list = f_list)  - 0.5*sum(r^2/M_diag) 
    log_prob0 <- f(theta = theta0, f_list = f_list) - 0.5*sum(r0^2/M_diag) 
    n <- (log(u) <= log_prob)
    s <- (log(u) < Delta_max + log_prob)
    alpha <- min(1, exp(log_prob - log_prob0))
    if(is.nan(alpha)) stop()
    if(is.na(s) || is.nan(s)){
      s <- 0
    }
    if(is.na(n) || is.nan(n)){
      n <- 0
    }
    return(list(theta_minus=theta, theta_plus=theta, theta=theta, r_minus=r,
                r_plus=r, s=s, n=n, alpha=alpha, n_alpha=1))
  } else{
    obj0 <- build_tree(theta, r, u, v, j-1, eps, theta0, r0, f, grad_f, f_list, M_diag)
    theta_minus <- obj0$theta_minus
    r_minus <- obj0$r_minus
    theta_plus <- obj0$theta_plus
    r_plus <- obj0$r_plus
    theta <- obj0$theta
    if(obj0$s == 1){
      if(v == -1){
        obj1 <- build_tree(obj0$theta_minus, obj0$r_minus, u, v, j-1, eps, theta0, r0, f, grad_f, f_list, M_diag)
        theta_minus <- obj1$theta_minus
        r_minus <- obj1$r_minus
      } else{
        obj1 <- build_tree(obj0$theta_plus, obj0$r_plus, u, v, j-1, eps, theta0, r0, f, grad_f, f_list, M_diag)
        theta_plus <- obj1$theta_plus
        r_plus <- obj1$r_plus
      }
      n <- obj0$n + obj1$n
      if(n != 0){
        prob <- obj1$n / n
        if(runif(1) < prob){
          theta <- obj1$theta
        }
      }
      s <- check_NUTS(obj1$s, theta_plus, theta_minus, r_plus, r_minus)
      alpha <- obj0$alpha + obj1$alpha
      n_alpha <- obj0$n_alpha + obj1$n_alpha
      
    } else{
      n <- obj0$n
      s <- obj0$s
      alpha <- obj0$alpha
      n_alpha <- obj0$n_alpha
    }
    if(is.na(s) || is.nan(s)){
      s <- 0
    }
    if(is.na(n) || is.nan(n)){
      n <- 0
    }
    return(list(theta_minus=theta_minus, theta_plus=theta_plus, theta=theta,
                r_minus=r_minus, r_plus=r_plus, s=s, n=n, alpha=alpha, n_alpha=n_alpha))
  }
}

hmc_deep <- function (theta, f, grad_f, f_list, epsilon = .1, L = 10, acc = 0) {
  p <- rnorm(length(theta), 0, 1) # independent standard normal variates
  # Make a half step for momentum at the beginning
  p_tilde <- p - epsilon*grad_f(theta = theta, f_list = f_list)/2
  # Alternate full steps for position and momentum
  for (i in 1: L){
    # Make a full step for the position
    theta_tilde = theta + epsilon*p_tilde
    # Make a full step for the momentum, except at end of trajectory
    if (i != L) p_tilde <- p_tilde - epsilon*grad_f(theta = theta, f_list = f_list)
  }
  # Make a half step for momentum at the end.
  p_tilde <- p_tilde - epsilon*grad_f(theta = theta, f_list = f_list)/2
  # Negate momentum at end of trajectory to make the proposal symmetric
  p_tilde <- -p_tilde
  # Evaluate potential and kinetic energies at start and end of trajectory
  log.prob_acc  = f(theta = theta, f_list = f_list)
  log.corr_acc  = sum(p^2) / 2
  log.prob_prop = f(theta = theta_tilde, f_list = f_list)
  log.corr_prop = sum(p_tilde^2) / 2 #IS THIS CORRECT??? 
  # Accept or reject the state at end of trajectory, returning either
  # the position at the end of the trajectory or the initial position
  if (log(runif(1)) < log.prob_acc - log.prob_prop + log.corr_acc - log.corr_prop) {
    theta = theta_tilde # accept
  }
  return(theta)
}


# Set of activation functions 
act.fc.set.all <- list(
  "uni" = list(
    "func" = function(x){
      x
    },
    "grad" = function(x){
      x^0
    }),
  "tanh" = list(
    "func" = function(x){
      tanh(x)},
    "grad" = function(x){
      1-(tanh(x))^2
    }),
  "relu" = list(
    "func" = function(x){
      (abs(x)+x)/2},
    "grad" = function(x){
      ifelse(x >= 0, 1, 0)
    }),
  "sigmoid" = list(
    "func" = function(x){
      (1 / (1 + exp(-x)))},
    "grad" = function(x){
      (1 / (1 + exp(-x)))*(1-(1 / (1 + exp(-x))))
    }),
  "leakyrelu" = list(
    "func" = function(x){
      ifelse(x >= 0, x, 0.01*x)},
    "grad" = function(x){
      ifelse(x >= 0, 1, 0.01)
    })
)

get.post_k <- function(theta, f_list){
# Posterior 
# Unpack list of arguments
k_draw    <- f_list$k_draw.nr
k.V       <- f_list$k.V
nr1       <- f_list$nr1
nr2       <- f_list$nr2
QQ        <- f_list$QQ
Q         <- f_list$Q
MM        <- f_list$MM
acf_draw  <- f_list$acf_draw
y         <- f_list$y
X         <- f_list$X
X.hat.nr  <- f_list$X.hat.nr
X.hat.wonr<- f_list$X.hat.wonr
b_draw.nr <- f_list$b_draw.nr
b_draw.wonr <- f_list$b_draw.wonr
sig2_draw <- f_list$sig2_draw
acf_set   <- f_list$acf_set 

fit.wonr <- X.hat.wonr[,,QQ]%*%b_draw.wonr
for (nn in 1:Q){
  Mlay <- MM[nn]
  X.hat.nr[,nr2,nn+1] <- acf_set[[acf_draw[nn]]][["func"]](X.hat.nr[,1:Mlay,nn]%*%as.matrix(k_draw.nr[1:Mlay,,nn]))
}
fit.nr <- as.matrix(X.hat.nr[,nr2,QQ])%*%b_draw.nr
  
loglik <- sum(dnorm(y, fit.wonr+fit.nr, sqrt(sig2_draw), log = TRUE))
logpr <- sum(dnorm(theta, 0, sqrt(k.V), log=TRUE))
logpost <- loglik + logpr
return(logpost)
}

get.post.grad_k <- function(theta, f_list = f_list){
  # Derivative of posterior (probably need to select qth column)
  # Unpack list of arguments
  k_draw    <- f_list$k_draw.nr
  k.V       <- f_list$k.V
  nr1       <- f_list$nr1
  nr2       <- f_list$nr2
  QQ  <- f_list$QQ
  Q <- f_list$Q
  MM   <- f_list$MM
  acf_draw  <- f_list$acf_draw
  y         <- f_list$y
  X         <- f_list$X
  X.hat.nr  <- f_list$X.hat.nr
  X.hat.wonr<- f_list$X.hat.wonr
  b_draw.nr <- f_list$b_draw.nr
  b_draw.wonr <- f_list$b_draw.wonr
  sig2_draw <- f_list$sig2_draw
  acf_set   <- f_list$acf_set 
  
  normalizer <- 1/sqrt(sig2_draw)
  fit.wonr <- (X.hat.wonr[,,QQ]*normalizer)%*%b_draw.wonr
  
  for (nn in 1:Q){
    Mlay <- MM[nn]
    X.hat.nr[,nr2,nn+1] <- acf_set[[acf_draw[nn]]][["func"]](X.hat.nr[,1:Mlay,nn]%*%as.matrix(k_draw[1:Mlay,,nn]))
  }
  fit.nr <- as.matrix(X.hat.nr[,nr2,QQ])%*%b_draw.nr
  
  yy <- (y*normalizer - fit.wonr)
  dhqdkq <- as.numeric(b_draw.nr)*acf_set[[acf_draw[nr1]]][["grad"]](X.hat.nr[,1:MM[nr1],nr1]%*%as.matrix(theta))*normalizer # Inner derivative of act. function
  dloglik <-  crossprod(X.hat.nr[,1:MM[nr1],nr1], (yy - fit.nr)*dhqdkq)
  
  dlogpr <- -1/k.V*theta      # Derivative of prior
  dlogpost <- dloglik + dlogpr
  return(dlogpost)
}

# HS prior
get.hs <- function(bdraw,lambda.hs,nu.hs,tau.hs,zeta.hs){
  k <- length(bdraw)
  if (is.na(tau.hs)){
    tau.hs <- 1   
  }else{
    tau.hs <- invgamma::rinvgamma(1,shape=(k+1)/2,rate=1/zeta.hs+sum(bdraw^2/lambda.hs)/2) 
  }
  
  lambda.hs <- invgamma::rinvgamma(k,shape=1,rate=1/nu.hs+bdraw^2/(2*tau.hs))
  
  nu.hs <- invgamma::rinvgamma(k,shape=1,rate=1+1/lambda.hs)
  zeta.hs <- invgamma::rinvgamma(1,shape=1,rate=1+1/tau.hs)
  
  ret <- list("psi"=(lambda.hs*tau.hs),"lambda"=lambda.hs,"tau"=tau.hs,"nu"=nu.hs,"zeta"=zeta.hs)
  return(ret)
}


