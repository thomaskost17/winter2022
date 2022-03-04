%%
 %  File: lorenz_feedback_linearize.m
 % 
 %  Author: Thomas Kost
 %  
 %  Date: 20 January 2022
 %  
 %  @brieffunction for lorenz attractor sim
 %
 function dx = lorenz_feedback_linearize(t,x,Beta)
 alpha1 = 10;
 u = -Beta(1)*(x(2)-x(1))-alpha1*x(1);
 dx =[...
    Beta(1)*(x(2) - x(1))+u;...
    x(1)*(Beta(2) - x(3)) - x(2);...
    x(1)*x(2) - Beta(3)*x(3);...
    ];
