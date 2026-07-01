### Code package: N. Hauzenberger, F. Huber, K. Klieber, & M. Marcellino (2025). Bayesian neural networks for macroeconomic analysis, *Journal of Econometrics*, 249(Part C):105843.

[**Publication (open access).**](https://doi.org/10.1016/j.jeconom.2024.105843)

### Data for the empirical application. 
In the empirical application, we use the popular FRED-MD database proposed in McCracken and Ng (2016). Our forecasting exercise focuses on the consumer price (CPIAUCSL) inflation rate as specified in Stock and Watson (1999), the month-on-month (m-o-m) growth rate of industrial production (INDPRO), and the m-o-m growth rate of employment (CE16OV). The sample ranges from January 1960 to December 2020. We assume that these three focus variables are an (unknown) function of the first lag of K = 120 economic and financial variables. Our focus is on short-term forecasting. Hence, we compute the one-month-ahead predictive distributions for our hold-out sample. For each target variable, we provide the data as a .rda files in the folder [`data`](./data/). 

### Estimation files. 
The file [`main_BNN.R`](main_BNN.R) allows to produce a single forecast for the main BNN specifications. In addition, the folder [`functions`](./functions/) provides several auxiliary functions, including efficient computation of the gradient, efficient evaluation of the posterior, and setup of the data matrices for the direct predictive regressions.
