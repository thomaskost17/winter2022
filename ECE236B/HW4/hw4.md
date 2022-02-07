# MATLAB Section

In this section we will be presenting the results and relevant code to problem 6 of homework 4. All sections are labeled accordingly.

# Results

the following text block shows the output from out MATLAB script. We have labeled the optimal value as `P_star` and the last recorded feasible input to solve the problem is labeled `approx x_opt`. Note the latter value does not solve for our ooptimal value but is simply close. The output is below:

```
---------------Solving Optimization--------------
This may take some time...
-------------------------------------------------
--------------------Results----------------------
P_star:: 0.04822
approx x_opt: 4.9106       5.023      5.2307
```

# Code

In this section all relevant code to perform bisection on the given quasiconvex problem is shown below.

```MATLAB
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
```
