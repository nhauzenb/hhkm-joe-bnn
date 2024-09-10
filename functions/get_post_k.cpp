#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace RcppArmadillo;
#include <RcppArmadilloExtensions/sample.h>

// [[Rcpp::export]]
double get_post_k(NumericVector theta, List f_list) {
  static Rcpp::Function asVector("as.vector");
  
  // 1) Unpack list of arguments
  // integers
  int nr1 = as<int>(f_list["nr1"]);
  int nr2 = as<int>(f_list["nr2"]);
  int QQ = as<int>(f_list["QQ"]);
  int Q = as<int>(f_list["Q"]);
  // vectors
  arma::vec      MM = f_list["MM"];
  arma::vec      acf_draw = f_list["acf_draw"];
  NumericVector  sig2_draw = f_list["sig2_draw"];
  NumericVector  k_V = f_list["k.V"];
  // matrices
  NumericVector  y = f_list["y"];
  arma::mat      b_draw_nr = f_list["b_draw.nr"];
  arma::mat      b_draw_wonr = f_list["b_draw.wonr"];
  arma::mat      X = f_list["X"];
  // cubes
  arma::cube     k_draw_nr = f_list["k_draw.nr"];
  arma::cube     X_hat_nr = f_list["X.hat.nr"];
  arma::cube     X_hat_nrcpp = X_hat_nr;
  arma::cube     X_hat_wonr = f_list["X.hat.wonr"];
  arma::cube     X_hat_wonrcpp = X_hat_wonr;
  // list
  List           acf_set = f_list["acf_set"];
  
  // 2) get fit of specific neuron (nr2)
  for (int nn = 0; nn < Q; nn++) {
    int Mlay = MM[nn];
    NumericMatrix X_hat_n = wrap(X_hat_nrcpp.slice(nn));
    NumericMatrix k_draw_n = wrap(k_draw_nr.slice(nn));
    arma::mat matX = X_hat_n(_,Range(0,Mlay-1));
    arma::mat matK1 = k_draw_nr.slice(nn);
    arma::mat matK = matK1.rows(0, Mlay-1);

    arma::mat X_hat_col = matX * matK;
    List acf_set_1 = acf_set[acf_draw[nn]-1];
    Function act_function = acf_set_1["func"];
    NumericMatrix X_hat_col_1 = act_function(X_hat_col);
    arma::mat X_hat_col_2 = Rcpp::as<arma::mat>(X_hat_col_1);
    X_hat_nrcpp.slice(nn+1).col(nr2-1) = X_hat_col_2;
  }

  arma::mat X_hat_final = X_hat_nrcpp.slice(QQ-1).col(nr2-1);
  arma::mat fit_nr = X_hat_final * b_draw_nr;
  
  // 3) get fit of all other neurons
  arma::mat matXwo = X_hat_wonrcpp.slice(QQ-1);
  arma::mat fit_wonr = matXwo * b_draw_wonr;
  
  arma::mat fit_sum = fit_nr + fit_wonr;
  NumericVector fit_final = wrap(fit_sum);
  
  // 4) get log likelihood
  double logLik = 0.0;
  double logPrior = 0.0;
  int ii1 = y.size();
  int ii2 = theta.size();
  
  for (int i = 0; i < ii1; i++) {
    double logProb = R::dnorm(y[i], fit_final[i], std::sqrt(sig2_draw[i]), true);
    logLik += logProb;
  }
  
  for (int i = 0; i < ii2; i++) {
    double logPrbp = R::dnorm(theta[i], 0, std::sqrt(k_V[i]), true);
    //sum(dnorm(theta, 0, sqrt(k.V[1:MM[nr1],nr2,nr1]), log=TRUE))
    logPrior += logPrbp;
  }
  
  double logpost = logLik + logPrior;
  
  return (logpost);

}

