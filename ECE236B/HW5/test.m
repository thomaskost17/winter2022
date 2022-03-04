n=20;
mu = logspace(-3,3,n);
T_delays = zeros(1,n);
Area = zeros(1,n);

for i=1:n
    cvx_begin GP
    variables w(6) T
    C = w;
    R = 1./w;
    minimize (sum(w) + mu(i)T)
    subject to
        Wmin <= w;
        w <= Wmax;

    (C(3) + C_load1)(R(1) + R(2) + R(3)) + C(2) * (R(1) + R(2)) + (C(1) + C(4) + C(5) + C(6) + C_load2 + C_load3) * R(1) <= T;
    (C(5) + C_load2)(R(1) + R(4) + R(5)) + C(4)(R(1) + R(4)) + (C(6) + C_load3)(R(1) + R(4)) + (C1 + C2 + C3 + C_load1)R(1) <= T;
    (C(6) + C_load3)(R(1) + R(4) + R(6)) + C4(R(1)+R(4)) + (C(5) + C_load2)(R(1)+R(4)) + (C(1) + C(2) + C(3) + C_load1)R(1) <= T;

    cvx_end

    T_delays(i) = max(T);
    Area(i) = sum(w);
end