// PRESS Non Parametric Regression

//  See https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46141.pdf  for more details


data {
  int <lower=0> N;                // Number of data points
  int <lower=0> J;                // Number of States
  int <lower=0> p;                // Dimension of input data
  vector[p] x[N];                 // input data
  vector[N] y;                    // output vector
  
  real<lower=0> prior_sigma;       // prior for the standard deviation of sigma parameter
  real<lower=0> prior_beta;       // prior for the standard deviation of beta parameters
  
}

parameters {

  matrix[max(1, J-1), p] beta_pre;              // Logistic regression parameters       
  real<lower=0> sigma;                          // standard deviation of errors`
  
}

transformed parameters{
  matrix[J, p] beta;                            // Logistic regression parameters       
  vector[J] y_bar;                              // state means  
  vector[N] mu;                                 // Mean PRESS regression value  
  matrix[N,J] W;              // Weight matrix
  vector[J] sigma_j_inv;              // sum of the jth column of W
  
  // fit beta to J-1 states and force final state to balance to zero
  if (J==1){
    beta = beta_pre;
  } else {
    beta[1:(J-1),1:p] = beta_pre[1:(J-1),1:p];
    
    for (pp in 1:p){
      beta[J, pp] = -sum(beta_pre[1:(J-1), pp]);
    }
  
  } 
  
  for (n in 1:N){
    W[n] = softmax(beta * x[n])';  // Logistic regression weights for each state
  }
  
  for (j in 1:J){
    sigma_j_inv[j] = 1 / sum(W[,j]);
  }
  
  // State Means
  y_bar =  diag_matrix(sigma_j_inv) * W' * y;
  
  // Mean Smoothing Function
  mu = W * y_bar;
  
}

model {
  
  // Priors
  to_vector(beta_pre) ~ cauchy(0,prior_beta);
  sigma ~ normal(0,prior_sigma);
  
  //Sampling statement
  y ~ normal (mu, sigma);
  
  
}

generated quantities{
  // vector[N] y_hat;                // predicted in-sample value
  // 
  // for (n in 1:N){
  //   y_hat[n] = normal_rng(mu[n], sigma);  // simulation step
  // }
}
