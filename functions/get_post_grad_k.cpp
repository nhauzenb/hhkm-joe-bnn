#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace RcppArmadillo;
#include <RcppArmadilloExtensions/sample.h>

// [[Rcpp::export]]
NumericVector get_post_grad_k(NumericVector theta, List f_list) {
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
  arma::vec theta_arma = Rcpp::as<arma::vec>(theta);
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
  NumericVector  normalizer = 1/Rcpp::sqrt(sig2_draw);
  arma::mat      y_norm = y * normalizer;
  
  k_draw_nr.slice(nr1-1).submat(0, 0, theta_arma.n_elem-1, 0) = theta_arma;
  
  // 2) get fit of specific neuron (nr2)
  arma::vec chain_grad(y.size(), arma::fill::ones);
  for (int nn = 0; nn < Q; nn++) {
    int Mlay = MM[nn];
    NumericMatrix X_hat_n = wrap(X_hat_nrcpp.slice(nn));
    NumericMatrix k_draw_n = wrap(k_draw_nr.slice(nn));
    arma::mat matX = X_hat_n(_,Range(0,Mlay-1));
    arma::mat matK1 = k_draw_nr.slice(nn);
    arma::mat matK = matK1.rows(0, Mlay-1);
    
    arma::mat X_hat_fit = matX * matK;
    List acf_set_1 = acf_set[acf_draw[nn]-1];
    Function act_function = acf_set_1["func"];
    NumericMatrix X_hat_col_1 = act_function(X_hat_fit);
    arma::mat X_hat_fit_save = Rcpp::as<arma::mat>(X_hat_col_1);
    X_hat_nrcpp.slice(nn+1).col(nr2-1) = X_hat_fit_save;
    
    if (nn+1 > nr1) {
      Function act_gradient = acf_set_1["grad"];
      NumericVector z_grad = act_gradient(X_hat_fit);
      arma::vec z_grad_arma = Rcpp::as<arma::vec>(z_grad);
      chain_grad %= z_grad_arma * matK(nr2-1, 0);
    }
  }
  arma::vec norm_arma = Rcpp::as<arma::vec>(normalizer);
  arma::mat X_hat_final = X_hat_nrcpp.slice(QQ-1).col(nr2-1);
  X_hat_final.col(0) %= norm_arma; 
  arma::mat fit_nr = X_hat_final * b_draw_nr;
  
  // 3) get fit of all other neurons
  
  int colX = X_hat_wonrcpp.slice(QQ-1).n_cols;
  for (int j = 0; j < colX; j++) {
    NumericVector XVector = wrap(X_hat_wonrcpp.slice(QQ-1).col(j));
    NumericVector Xnorm(normalizer.size());
    for (size_t i = 0; i < normalizer.size(); i++) {
      Xnorm[i] = XVector[i] * normalizer[i];
    }
    arma::mat Xnorm_save = Rcpp::as<arma::vec>(Xnorm);
    X_hat_wonrcpp.slice(QQ-1).col(j) = Xnorm_save;
  }
  
  arma::mat fit_wonr = X_hat_wonrcpp.slice(QQ-1) * b_draw_wonr;
  arma::mat fit_sum = fit_nr + fit_wonr;
  NumericVector fit_final = wrap(fit_sum);
  
  
  // 4) get posterior: log likelihood and prior
  int Mlay = MM[nr1-1];
  arma::mat  yy = (y_norm - fit_wonr);
  
  arma::mat  matTheta = theta;
  arma::mat  matXhat1  = X_hat_nrcpp.slice(nr1-1);
  arma::mat  matXhat = matXhat1.cols(0, Mlay-1);
  arma::mat  X_hat_mult = matXhat * matTheta;
  
  List acf_set_1 = acf_set[acf_draw[nr1-1]-1];
  Function act_grad = acf_set_1["grad"];
  NumericVector X_hat_theta = act_grad(X_hat_mult);
  NumericVector b_draw_der = wrap(b_draw_nr);
  
  // Inner derivative of act. function
  NumericVector dhqdkq(X_hat_theta.size());
  for (size_t i = 0; i < X_hat_theta.size(); i++) {
    dhqdkq[i] = b_draw_der[0] * chain_grad[i] * X_hat_theta[i] * normalizer[i];
  }
  // multiply error with inner derivative
  NumericVector err_der(dhqdkq.size());
  for (size_t i = 0; i < dhqdkq.size(); i++) {
    err_der[i] = (yy - fit_nr)[i] * dhqdkq[i];
    
  }
  
  arma::mat materr =Rcpp::as<arma::vec>(err_der);
  arma::mat crossProd = trans(matXhat) * materr;
  
  NumericVector dlogLik = wrap(crossProd);
  
  NumericVector dlogpost(dlogLik.size());
  for (int i = 0; i < dlogLik.size(); i++) {
    dlogpost[i] = dlogLik[i] - theta[i]/k_V[i];
  }
  
  return (dlogpost);
  
}

