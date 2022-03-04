%%
 %  File: hw5.m
 % 
 %  Author: Thomas Kost
 %  
 %  Date: 10 February 2022
 %  
 %  @brief homework 5 matlab spacecraft landing
 %
 clc,clear all,close all;
 
 %% 2C:
 
 % Import data
 spacecraft_landing_data
 % Define DTSS propogation
 A = [eye(3),eye(3)*h; zeros(3,3),eye(3)];
 std_basis = eye(3);
 B = [eye(3)*(h^2/(2*m)), -std_basis(:,3)*(g*h^2)/2;
      eye(3)*(h/m), -std_basis(:,3)*(g*h);];
prop = zeros(6,K*4);
 for i = 0:K-1
     prop(:,4*i+1:4*i+4) = (A^(i))*B;
 end
 % Optimal fuel calculation
disp('-------- Optimal Fuel Solution-------');

cvx_begin 
variable f(3,K)
minimize(gamma*h*sum(norms(f,2)))
subject to
    expression x(6,K+1)
        x(:,1) = [p0;v0];
        for i = 2:K+1
            x(:,i) = A*x(:,i-1) + B*[f(:,i-1);1];
            x(3,i) >= alpha*norm(x(1:2,i),2)
        end
    zeros(6,1) == x(:,K+1);
    norms(f,2) <= Fmax*ones(1,K)
cvx_end
disp('-------- solved, making plots -------');
disp(' ');

p = x(1:3,:);
v = x(4:6,:);

s = linspace(-40,55,30); y = linspace(0,55,30);
[X,Y] = meshgrid(s,y);
Z = alpha*sqrt(X.^2+Y.^2);
min_fuel = figure; colormap autumn; surf(X,Y,Z);
axis([-40,55,0,55,0,105]);
grid on; hold on;

plot3(p(1,:),p(2,:),p(3,:),'b','linewidth',1.5);
quiver3(p(1,1:K),p(2,1:K),p(3,1:K),...
        f(1,:),f(2,:),f(3,:),0.3,'k','linewidth',1.5);
saveas(min_fuel, "2c_min_fuel.jpg")
min_fuel_profile =figure;
plot(norms(f))
xlabel("time_steps");
ylabel("||f||")
saveas(min_fuel_profile, "min_fuel_bang.jpg")
% Compute minimum time optimization
disp('-------- Optimal time Solution-------');
min_steps = K;
for j =K:-1:1
    cvx_begin quiet
    variable f(3,j)
    minimize 0
    subject to
        expression x(6,j+1)
            x(:,1) = [p0;v0];
            for i = 2:j+1
                x(:,i) = A*x(:,i-1) + B*[f(:,i-1);1];
                x(3,i) >= alpha*norm(x(1:2,i),2)
            end
        zeros(6,1) == x(:,j+1);
        norms(f,2) <= Fmax*ones(1,j)
    cvx_end
    
    if cvx_optval == inf
        break
    else
        min_steps = j;
        x_valid = x;
        f_valid = f;
    end
end
disp('--------------- solved --------------');
disp(['Minimum Steps:  ', num2str(min_steps)]);
s = linspace(-40,55,30); y = linspace(0,55,30);
[X,Y] = meshgrid(s,y);
Z = alpha*sqrt(X.^2+Y.^2);
min_time = figure; colormap autumn; surf(X,Y,Z);
axis([-40,55,0,55,0,105]);
grid on; hold on;
p = x_valid(1:3,:);
v = x_valid(4:6,:);

plot3(p(1,:),p(2,:),p(3,:),'b','linewidth',1.5);
quiver3(p(1,1:min_steps),p(2,1:min_steps),p(3,1:min_steps),...
        f_valid(1,:),f_valid(2,:),f_valid(3,:),0.3,'k','linewidth',1.5);
saveas(min_time, "2c_min_time.jpg");
    figure
plot(norms(f_valid))
disp(' ');


%% 5a:
% Store Relevant values
C_load = [1.5; 1; 5];
W_min = 0.1;
W_max = 10;
k0 =1;

disp('--------Plotting Area vs. Delay Relation-------');
disp('--------Plots Generated-------');
w_fix = linspace(W_min,W_max, 1000)';
area_delay_plot =figure;
A = 6*w_fix;
C = k0*w_fix;
R = 1./w_fix;
T = max((C+C_load(1)).*(3.*R) +2.*C.*R +(4.*C+C_load(2)+C_load(3)).*R,...
        max((C+C_load(2)).*3.*R +2.*C.*R+(C+C_load(3)).*2.*R +(3.*C+C_load(1)).*R,...
        (C+C_load(3)).*3.*R +C.*2.*R+(3.*C +C_load(1)).*R+(C+C_load(2)).*2.*R));
plot(A',T')
xlabel("Area");
ylabel("Delay");
title("Constant Widths")
saveas(area_delay_plot, "5a.jpg")
disp (' ')
%% 5b:
disp('-------- Optimal widths Solution-------');
mu = logspace(-3,3,40);
opts = zeros(length(mu),1);
T_opts = zeros(length(mu),1);
A_opts = zeros(length(mu),1);
for k=1:40
    cvx_begin gp quiet
    variables w(6) T(3)
    minimize(sum(w)+mu(k)*max(T))
    subject to
        C = k0*w;
        R = 1./w;
        T(1) >= (C(3) + C_load(1))*(R(1)+R(2)+R(3))+C(2)*(R(1)+R(2)) +(C(1)+C(4)+C(5)+C(6)+C_load(2)+C_load(3))*R(1)
        T(2) >= (C(5) + C_load(2))*(R(1)+R(4)+R(5)) +C(4)*(R(1)+R(4))+(C(6)+C_load(3))*(R(1)+R(4))+(C(1)+C(2)+C(3)+C_load(1))*R(1)
        T(3) >= (C(6)+C_load(3))*(R(1)+R(4)+R(6)) + C(4)*(R(1)+R(4)) +(C(1)+C(2)+C(3)+C_load(1))*R(1) + (C(5)+C_load(2))*(R(1)+R(4))
        w <= W_max
        w >= W_min
    cvx_end
    opts(k) = cvx_optval;
    A_opts(k) = sum(w);
    T_opts(k) = max(T);
    if(~mod(k,5))
        disp([num2str(k), ' itterations complete...']);
    end
end
opt_w =figure
plot(A_opts,T_opts)
xlabel("Optimal A")
ylabel("Optimal T")
title("Optimal Widths")
saveas(opt_w, "5b.jpg");
