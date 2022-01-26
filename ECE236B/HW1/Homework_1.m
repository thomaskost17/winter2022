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