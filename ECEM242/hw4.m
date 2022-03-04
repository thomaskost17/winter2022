%%
 %  File: hw4.m
 % 
 %  Author: Thomas Kost
 %  
 %  Date: 10 February 2022
 %  
 %  @brief homework 4 feedback linearization and backstepping controller
 %  comparrison for Lorenz attractor
 %
 clc,clear all,close all;
 %% Define system
 sigma = 10;
 rho = 28;
 beta = 8/3;
 Beta = [sigma; rho; beta];
 x_0 = [0, 1, 20];
 
 %% Simulate System
 dt = 0.001;
 tspan = [0:dt:10];
 [t,x] = ode45(@(t,x)lorenz_backstepped(t,x,Beta), tspan, x_0);
 [t1,x1] = ode45(@(t1,x1)lorenz_feedback_linearize(t1,x1,Beta), tspan, x_0);
  
 lambda = 2*Beta(2)*Beta(1)+1;
 u = (-Beta(1) + lambda/(2*Beta(2))- 1/(2*Beta(2)))*x(:,2) +(Beta(1) +0.5 -lambda)*x(:,1)-...
     x(:,1).*x(:,3)/(2*Beta(2));
 
 alpha1 = 10;
 u1 = -Beta(1)*(x1(:,2)-x1(:,1))-alpha1*x1(:,1);
 control_input_plot = figure();
 hold on;
 plot(t,u);
 plot(t1,u1);
 hold off;
 title("Control Inputs");
 xlabel('Time(s)');
 ylabel('Magnitude (unitless)');
 legend('Backstepping Controller', 'Feedback Linearization Controller');
 saveas(control_input_plot, 'control_inputs.jpg');
 trajectory_plot = figure();
 plot3(x(:,1),x(:,2),x(:,3),x1(:,1),x1(:,2),x1(:,3))
 legend('Backstepping', 'Feedback Linearization')
 xlabel('x(t)')
 ylabel('y(t)')
 zlabel('z(t)')
 title('Backstepping and Feed Back linearization')
 saveas(trajectory_plot, "trajectory_plot_hw4.jpg");
    uncontrolled = figure();
 [t,x2] = ode45(@(t,x)lorenz(t,x,Beta), tspan, x_0);
 plot3(x2(:,1),x2(:,2),x2(:,3),x(:,1),x(:,2),x(:,3),x1(:,1),x1(:,2),x1(:,3))
 xlabel('x(t)')
 ylabel('y(t)')
 zlabel('z(t)')
 title("Uncontrolled Trajectory vs. controlled ");
  legend('Free Resoponse','Backstepping', 'Feedback Linearization')

  saveas(uncontrolled, "uncontrolled_trajectory_plot_hw4.jpg");
