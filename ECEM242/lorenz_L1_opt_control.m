%%
 %  File: lorenz_L1_opt_control.m
 % 
 %  Author: Thomas Kost
 %  
 %  Date: 20 January 2022
 %  
 %  @brieffunction lorenz attractor diff eq with L1 control
 %
  function dx = lorenz_L1_opt_control(t,x,Beta)
     % Constants 
     lambda3 = Beta(4);
     e = x(1)-x(2)/(2*Beta(2));
     V = 0.5*x(2)^2+0.5*x(3)^2 +0.5*e^2;
    
     % Calcualte Optimal Input
%      cvx_precision low
%      cvx_begin quiet
%      variable u(1)
%      minimize(norm(u,1))
%      subject to
%             % Stability Contraints
%             -Beta(3)*x(3)^2-0.5*x(2)^2+ x(2)*Beta(2)*e + e*(Beta(1)*(x(2)-x(1))...
%                 -(x(1)*(Beta(2)-x(3))-x(2))/(2*Beta(2)))...
%                 + e*u <= -lambda3*V;
%             % Actuator Contraints
%      cvx_end
    options = optimoptions('fmincon','Display', 'off');
     fun = @(u)norm(u,1);
     b = -(-Beta(3)*x(3)^2-0.5*x(2)^2+ x(2)*Beta(2)*e + e*(Beta(1)*(x(2)-x(1))...
                -(x(1)*(Beta(2)-x(3))-x(2))/(2*Beta(2))))-lambda3*V;
     u = fmincon(fun, 1000,e,b,[],[],[],[],[], options);
%       u
%       V
%       t
     % Return x_dot = f(x)+g(x)*u
     dx =[...
        Beta(1)*(x(2) - x(1))+u;...
        x(1)*(Beta(2) - x(3)) - x(2);...
        x(1)*x(2) - Beta(3)*x(3);...
        ];
