%%
 %  File: Homework_3.m
 % 
 %  Author: Thomas Kost
 %  
 %  Date: 25 January 2022
 %  
 %  @brief homework 3 matlab problem concerning optimal fuel control
 %
 clear all, clc, close all;

 % Problem Data
 A = [-1, 0.4, 0.8;
       1,   0,   0;
       0,   1,   0];
 b = [1; 0; 0.3];
 x_des = [7; 2; -6];
 N = 30;
 x_0 = [0;0;0];
 prop = zeros(3,N);
 for i = 1:N
     prop(:,i) = (A^(i-1))*b;
 end
     prop = fliplr(prop);
 % Perform optimization
cvx_begin
    variable u(N)
    minimize(sum(max(abs(u),2*abs(u)-1)))
    subject to
        prop*u == x_des
cvx_end
input = figure()
plot([0:N-1],u);
xlabel("time step (unitless)");
ylabel("Actualtor Signal (unitless)");
title("Optimal Actuator Signal for Minimal Fuel Control");
saveas(input, "optimal_input.jpg");