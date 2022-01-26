%%
 %  File: Homework_1.m
 % 
 %  Author: Thomas Kost
 %  
 %  Date: 20 January 2022
 %  
 %  @brief homework 1 matlab simulation of continuity of solutions bound
 %
 clear all, clc, close all;
 %% Define system
 sigma = 10;
 rho = 28;
 beta = 8/3;
 Beta = [sigma; rho; beta];
 %% Define initial conditions
 delta = 0.1;
 x_0 = [0, 1, 20];
 x_1 = x_0 - [delta,0,0];
 
 dt = 0.001;
 tspan = [0:dt:5];
 [t,x] = ode45(@(t,x)lorenz(t,x,Beta), tspan, x_0);
 [t1,x1] = ode45(@(t1,x1)lorenz(t1,x1,Beta), tspan, x_1);
 trajectory_plot = figure();
 plot3(x(:,1),x(:,2),x(:,3),x1(:,1),x1(:,2),x1(:,3))
 xlabel('x(t)')
 ylabel('y(t)')
 zlabel('z(t)')
 title("Trajectory plot");
 saveas(trajectory_plot, "trajectory_plot_hw1.jpg");
 max_disp = max(abs(x-x_0)*ones(3,1)); %one norm
 epsillon = ceil(max_disp);
 L = max([2*sigma, abs(rho-x_0(3))+abs(x_0(1))+1+2*epsillon, abs(x(2)) +abs(x(1))+beta+2*epsillon]);
 
 % Compute distances
 bound = norm(x_0-x_1,1)*exp(L*t);
 dist = abs(x-x1)*ones(3,1);
 bound_plot = figure();
 hold on;
 plot(t,bound);
 plot(t,dist);
 xlim([0 0.005]);
 legend(["bound", "distance of unique solutions"]);
 xlabel("t");
 ylabel("Distance");
 title("Comparrison of upper bound to actual distance");
 hold off;
 saveas(bound_plot, "bound_plot_hw1.jpg");