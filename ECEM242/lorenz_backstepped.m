%%
 %  File: lorenz_backstepped.m
 % 
 %  Author: Thomas Kost
 %  
 %  Date: 20 January 2022
 %  
 %  @brieffunction for lorenz attractor sim
 %
 function dx = lorenz_backstepped(t,x,Beta)
 lambda = 2*Beta(2)*Beta(1)+1;
 u = (-Beta(1) + lambda/(2*Beta(2))- 1/(2*Beta(2)))*x(2) +(Beta(1) +0.5 -lambda)*x(1)-...
     x(1)*x(3)/(2*Beta(2));
 dx =[...
    Beta(1)*(x(2) - x(1))+u;...
    x(1)*(Beta(2) - x(3)) - x(2);...
    x(1)*x(2) - Beta(3)*x(3);...
    ];
