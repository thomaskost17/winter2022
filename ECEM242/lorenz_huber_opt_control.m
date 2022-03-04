%%
 %  File: lorenz_huber_opt_control.m
 % 
 %  Author: Thomas Kost
 %  
 %  Date: 20 January 2022
 %  
 %  @brieffunction lorenz attractor diff eq with L1 control
 %
  function dx = lorenz_huber_opt_control(t,x,Beta)
     % Constants 
     lambda3 = Beta(4);
     e = x(1)-x(2)/(2*Beta(2));
     V = 0.5*x(2)^2+0.5*x(3)^2 +0.5*e^2;
     
     % Calcualte Optimal Input
     cvx_begin quiet
     variable u(1)
     minimize(huber(u))
     subject to
            % Stability Contraints
            -Beta(3)*x(3)^2-0.5*x(2)^2 + e*(Beta(1)*(x(2)-x(1))...
                -(x(1)*(Beta(2)-x(3))-x(2))/(2*Beta(2)))...
                + e*u <= -lambda3*V;
            % Actuator Contraints
     cvx_end
     % Return x_dot = f(x)+g(x)*u
     dx =[...
        Beta(1)*(x(2) - x(1))+u;...
        x(1)*(Beta(2) - x(3)) - x(2);...
        x(1)*x(2) - Beta(3)*x(3);...
        ];
