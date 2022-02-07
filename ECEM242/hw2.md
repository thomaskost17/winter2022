# MATLAB Code

For this homework, MATLAB was used to perform the linearization of the lorentz attrractor. It should be noted that to calculate the eigenvalues at different equilibria, the desired equilibria would need to be typed into the code--however, this is trivial and thus left to anyone with interest. It is currently shown for a non-trivial equilibrium. The code is shown in its entirety below

# Code
```MATLAB
%%
 %  File: hw2.m
 % 
 %  Author: Thomas Kost
 %  
 %  Date: 20 January 2022
 %  
 %  @brief homework 2 matlab stability investigation
 %
 clear all, clc, close all;
  %% Define system
 sigma = 10;
 rho = 28;
 beta = 8/3;
 Beta = [sigma; rho; beta];
 
 alt_eq = sqrt(beta*(rho-1));
 x_eq = [-alt_eq;-alt_eq;rho-1];
 
 Df = [-sigma,rho-x_eq(3), x_eq(2);
        sigma,         -1, x_eq(1);
            0,   -x_eq(1),   -beta;];
 eig(Df)
 ```