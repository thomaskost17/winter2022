 %  File: hw8 .m
 % 
 %  Author: Thomas Kost
 %  
 %  Date: 10 February 2022
 %  
 %  @brief homework 5 creating family of optimal controllers 
 %  from existing Control Lyapanov Function(CLF)
 %
  clc,clear all,close all;
 %% Define system
 sigma = 10;
 rho = 28;
 beta = 8/3;
 Beta = [sigma; rho; beta];
 x_0 = [0, 1, 20];
 
  %% Simulate System
 lambda3 = 1;
 Beta = [Beta;lambda3];
 dt = 0.1;
 tspan = [0 2.5];
%  options = odeset('RelTol', 1e-2, 'AbsTol', 1e-4);
 tic;
 disp("Calcualting L1 Optimal Controller");
 [t,x] = ode23(@(t,x)lorenz_L1_opt_control(t,x,Beta), tspan, x_0);
 toc
 disp(' ');
%   disp("Calcualting L_inf Optimal Controller");
%   tic;
%  [t1,x1] = ode45(@(t1,x1)lorenz_L_inf_opt_control(t1,x1,Beta), tspan, x_0);
%   toc
%  disp(' ');
%   disp("Calcualting Huber Optimal Controller");
%   tic;
%  [t2,x2] = ode45(@(t2,x2)lorenz_huber_opt_control(t2,x2,Beta), tspan, x_0);
%   toc
%  disp(' ');
   disp("Calcualting Backstepping Controller");
  tic;
  [t3,x3] = ode45(@(t3,x3)lorenz_backstepped(t3,x3,Beta(1:3)), tspan, x_0);
  toc
 disp(' ');
 %Calcualte inputs afterwards because ODE45 is a pain
 
 %% Plot Results
 disp("Plotting Results");
 trajectory_plot = figure();
 plot3(x(:,1),x(:,2),x(:,3),x3(:,1),x3(:,2),x3(:,3) )
  legend('L_{1}', 'Backstepping');

%  plot3(x(:,1),x(:,2),x(:,3),x1(:,1),x1(:,2),x1(:,3), x2(:,1),x2(:,2),x2(:,3),x3(:,1),x3(:,2),x3(:,3))
%   legend('L1', 'L_{\infty}', 'Huber', 'Backstepping');

 xlabel('x(t)')
 ylabel('y(t)')
 zlabel('z(t)')
 saveas(trajectory_plot, "optimal comparrisons.jpg");