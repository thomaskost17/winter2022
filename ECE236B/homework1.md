## MATLAB Section

The following sections describe the code produced for problem 5 and the corresponding output. Note that the code is divided into sections relevant to each question. The output is similarly annotated.


## Code
```MATLAB
%%
 %  File: Homework_1.m
 % 
 %  Author: Thomas Kost
 %  
 %  Date: 7 January 2022
 %  
 %  @brief homework 1 matlab problem concerning optimal lighting of a
 %  surface
 %
 clear all, clc, close all;
 
 % Import data
 illumdata
 %% 5a: Least Squares
  disp('PERFORMING LEAST SQUARES');
 disp('------------------------------------');
 sz_A = size(A);
 b =  ones(sz_A(1),1);
 x = A\b;
 x(x<0) = 0;
 x(x>1) = 1;
 disp(['p = ', num2str(x')]);
 disp(' ');
 
 %% 5b: Regularized Least Squares 
 disp('PERFORMING REGULARIZED LEAST SQUARES');
 disp('------------------------------------');
 rho =0;
 d_rho =0.001;
 scaled = false;
 x_reg = zeros(sz_A(2));
 while ~scaled 
     A_prime = [A;sqrt(rho)*eye(sz_A(2))];
     b_prime = [b;sqrt(rho)*0.5*ones(sz_A(2),1)];
     x_reg = A_prime \b_prime;
     scaled = logical(prod(x_reg>=0)*prod(x_reg<=1));
     if ~scaled
         rho = rho + d_rho;
     end

 end
 disp(['rho: ', num2str(rho)]);
 disp(['p = ', num2str(x_reg')]);
 disp(' ');

 %% 5c: Chebychev Approximation
 disp('PERFORMING CHEBYCHEV APPROXIMATION');
 disp('------------------------------------');
 A_lp = [ A, -ones(sz_A(1),1);
         -A, -ones(sz_A(1),1);
          eye(sz_A(2)),zeros(sz_A(2),1);
          -eye(sz_A(2)),zeros(sz_A(2),1);
        ];
 b_lp = [b;-b; ones(sz_A(2),1); zeros(sz_A(2),1)];
 f = [zeros(sz_A(2),1);1];
 x_lin = linprog(f,A_lp,b_lp);
 x_lin = x_lin(1:sz_A(2));
 disp(['p = ', num2str(x_lin')]);
 disp(' ');
    
 %% 5d: Exact Solution
 disp('CALCULATING EXACT SOLUTION');
 disp('------------------------------------');
cvx_begin
    variable p(sz_A(2))
    minimize(max(max(inv_pos(A*p),A*p)))
    subject to
         p <= ones(sz_A(2),1)
        -p <= zeros(sz_A(2),1)
cvx_end
disp(' ')
disp(['p = ', num2str(p')]);
```

## Outputs

## 5a
```
PERFORMING LEAST SQUARES
------------------------------------
p = 1  0  1  0  0  1  0  1  0  1
```
## 5b
```
PERFORMING REGULARIZED LEAST SQUARES
------------------------------------
rho: 0.219
p = 0.50042     0.47769    0.083304  0.00022527     0.45608     0.43543     0.45971     0.43072     0.40343     0.45264
```
## 5c
```
PERFORMING CHEBYCHEV APPROXIMATION
------------------------------------

Optimal solution found.

p = 1      0.1165           0           0           1           0           1    0.024901           0           1
```
## 5d
The following is the output of cvx and the resulting optimum vector `p` shown below.
```
CALCULATING EXACT SOLUTION
------------------------------------
 
Calling SDPT3 4.0: 140 variables, 51 equality constraints
   For improved efficiency, SDPT3 is solving the dual problem.
------------------------------------------------------------

 num. of constraints = 51
 dim. of sdp    var  = 40,   num. of sdp  blk  = 20
 dim. of linear var  = 80
*******************************************************************
   SDPT3: Infeasible path-following algorithms
*******************************************************************
 version  predcorr  gam  expon  scale_data
   HKM      1      0.000   1        0    
it pstep dstep pinfeas dinfeas  gap      prim-obj      dual-obj    cputime
-------------------------------------------------------------------
 0|0.000|0.000|2.0e+02|6.9e+00|1.7e+04| 1.000000e+02  0.000000e+00| 0:0:00| chol  1  1 
 1|0.926|0.815|1.5e+01|1.3e+00|1.9e+03| 1.190120e+02 -3.720242e+01| 0:0:00| chol  1  1 
 2|0.885|1.000|1.7e+00|4.6e-03|2.9e+02| 1.221977e+02 -4.485628e+01| 0:0:00| chol  1  1 
 3|0.988|1.000|2.0e-02|4.6e-04|2.5e+01| 7.958638e-01 -2.370968e+01| 0:0:00| chol  1  1 
 4|0.912|0.867|1.8e-03|4.2e-03|4.1e+00|-5.228834e-01 -4.550569e+00| 0:0:00| chol  1  1 
 5|1.000|0.671|2.9e-10|1.7e-03|2.8e+00|-3.981550e-01 -3.229143e+00| 0:0:00| chol  1  1 
 6|1.000|0.936|2.1e-10|1.1e-04|6.4e-01|-9.678560e-01 -1.609940e+00| 0:0:00| chol  1  1 
 7|1.000|1.000|1.5e-11|4.6e-08|2.9e-01|-1.260326e+00 -1.547825e+00| 0:0:00| chol  1  1 
 8|1.000|1.000|2.0e-11|4.6e-09|6.5e-02|-1.377256e+00 -1.441997e+00| 0:0:00| chol  1  1 
 9|0.937|0.980|1.6e-11|5.5e-10|1.5e-02|-1.417143e+00 -1.432303e+00| 0:0:00| chol  1  1 
10|0.989|1.000|7.0e-13|4.9e-11|1.9e-03|-1.428070e+00 -1.429966e+00| 0:0:00| chol  1  1 
11|0.998|0.958|2.5e-11|7.5e-12|1.1e-04|-1.429627e+00 -1.429735e+00| 0:0:00| chol  1  1 
12|0.999|0.997|2.8e-12|1.5e-12|2.1e-06|-1.429712e+00 -1.429714e+00| 0:0:00| chol  1  1 
13|1.000|1.000|6.0e-11|1.0e-12|2.7e-08|-1.429714e+00 -1.429714e+00| 0:0:00|
  stop: max(relative gap, infeasibilities) < 1.49e-08
-------------------------------------------------------------------
 number of iterations   = 13
 primal objective value = -1.42971383e+00
 dual   objective value = -1.42971386e+00
 gap := trace(XZ)       = 2.75e-08
 relative gap           = 7.11e-09
 actual relative gap    = 7.07e-09
 rel. primal infeas (scaled problem)   = 5.98e-11
 rel. dual     "        "       "      = 1.00e-12
 rel. primal infeas (unscaled problem) = 0.00e+00
 rel. dual     "        "       "      = 0.00e+00
 norm(X), norm(y), norm(Z) = 1.4e+00, 2.9e+00, 1.0e+01
 norm(A), norm(b), norm(C) = 1.6e+01, 2.0e+00, 2.4e+01
 Total CPU time (secs)  = 0.48  
 CPU time per iteration = 0.04  
 termination code       =  0
 DIMACS: 6.0e-11  0.0e+00  4.3e-12  0.0e+00  7.1e-09  7.1e-09
-------------------------------------------------------------------
 
------------------------------------------------------------
Status: Solved
Optimal value (cvx_optval): +1.42971
 
TRUE OPTIMUM:
p = 1      0.2023  1.1778e-08  7.8265e-09           1  4.5358e-07           1     0.18816  8.6109e-08           1
```
