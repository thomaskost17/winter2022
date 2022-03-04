%%
 %  File: hw8 .m
 % 
 %  Author: Thomas Kost
 %  
 %  Date: 10 February 2022
 %  
 %  @brief homework 8 matlab problems concerning nonlinear measurement
 %  noise
 %
 clc,clear all,close all;
 nonlin_meas_data;
disp('Determining Measurement Nonlinearity');
disp('------------------------------------');
disp(' ');
disp('Beginning Optimization ...');
cvx_begin quiet
variables z(m) x(n)
maximize(-(m/2)*log(2*pi*sigma^2)-(1/(2*sigma^2))*sum((A*x-z).^2))
subject to
    for i = 1:m-1
        y(i+1)-y(i) >= alpha*(z(i+1)-z(i));
        y(i+1)-y(i) <= beta*(z(i+1)-z(i));

    end
cvx_end
disp(' ');
disp(['Optimal x : ', num2str(x')]);
nl_plot = figure();
plot(z,y);
title("Measurement Nonlinearity", 'Interpreter', 'latex');
xlabel('$\hat{z}_{ml}$', 'Interpreter', 'latex');
ylabel('y', 'Interpreter', 'latex');
saveas(nl_plot, 'nl_plot.jpg');
