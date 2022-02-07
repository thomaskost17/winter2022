%%
 %  File: hw4.m
 % 
 %  Author: Thomas Kost
 %  
 %  Date: 3 February 2022
 %  
 %  @brief homework 4 matlab problem concerning image triangulation
 %
 clc,clear all,close all;
 
 %% Define problem setup
 N=4;
 P1 = [eye(3),ones(3,1)];
 P2 = [1  0 0 0;
       0  0 1 0;
       0 -1 0 10;];
 P3 = [ 1  1 1 -10;
       -1  1 1  0;
       -1 -1 1  10;];
 P4 = [0  1 1 0;
       0 -1 1 0;
      -1  0 0 10;];

y1 = [0.98;0.93];
y2 = [1.01;1.01];
y3 = [0.95;1.05];
y4 = [2.04;0];

%% Solve Optimization
epsilon = 1e-4;
l=0;
u=1000;
disp('---------------Solving Optimization--------------')
disp('This may take some time...')

while u-l >= epsilon
    t = (u+l)/2;
    cvx_begin quiet
        variable x(3)
        minimize(0)
        subject to
            norm([eye(2),-y1]*P1*[x;1])-t*[0 0 1]*P1*[x;1] <=0
            norm([eye(2),-y2]*P2*[x;1])-t*[0 0 1]*P2*[x;1] <=0
            norm([eye(2),-y3]*P3*[x;1])-t*[0 0 1]*P3*[x;1] <=0
            norm([eye(2),-y4]*P4*[x;1])-t*[0 0 1]*P4*[x;1] <=0
    cvx_end
    if cvx_optval == inf
        l=t;
    else
        u=t;
        last_val_x = x;
    end
end
disp('-------------------------------------------------')

%% Report Results
disp('--------------------Results----------------------')
disp(['P_star:: ', num2str(t)]);
disp(['approx x_opt: ', num2str(last_val_x')])
