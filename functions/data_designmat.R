
freq <- 12
load(paste0(w.dir, "data/Macro1_", target, "_Monthly_nhor1.rda"))

lag.slct <- strsplit(target, "_")[[1]][[1]]
var.slct <- info.sets[[info]]

Y <- window(Y, end = hout, frequency = freq)
X <- window(X, end = hout, frequency = freq)
TT <- nrow(Y)
Y.lag <- X[,lag.slct, drop = F]
X <- X[,var.slct,drop = F]

Xho <- X[TT,,drop = F]
yho <- Y[TT,,drop = F]

Nho <- nrow(yho)
y <- Y[1:(TT-1),,drop = F]
X <- X[1:(TT-1),,drop = F]

K <- ncol(X)
N <- nrow(y)

if(stdz){
  y.mu <- mean(y)
  y.sd <- sd(y)
  X.mu <- apply(X, 2, mean)
  X.sd <- apply(X, 2, sd)
  
  # Standardize y
  y <- (y - y.mu)/y.sd
  yho <- (yho - y.mu)/y.sd
  # Standardize X
  X <- (X - t(matrix(X.mu, K, dim(X)[1])))/t(matrix(X.sd, K, dim(X)[1]))
  Xho <- (Xho - t(matrix(X.mu, K, Nho)))/t(matrix(X.sd, K, Nho))
  
} else {
  y.mu <- 0
  y.sd <- 1
}

DGP.slct <- list("X" = X, "y" = y, "yho" = yho, "Xho" = Xho, 
                 "y.mu" = y.mu, "y.sd" = y.sd, "stdz" = stdz,
                 "ID" = var.slct, "info" = info)

