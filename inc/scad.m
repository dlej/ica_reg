function [ y, dy ] = scad( x, lambda, alpha )
%SCAD SCAD penalty and derivative
%   Detailed explanation goes here
    
    [m,n] = size(x);
    y = zeros(m,n);
    dy = y;
    
    case1 = abs(x) <= lambda;
    case2 = logical((abs(x) <= alpha*lambda) .* ~case1);
    case3 = logical(~case1 .* ~case2);
    
    y(case1) = lambda*abs(x(case1));
    dy(case1) = lambda*sign(x(case1));
    
    y(case2) = (-x(case2).^2 + 2*alpha*lambda*abs(x(case2)) - lambda^2)/(2*(alpha-1));
    dy(case2) = (-2*x(case2) + 2*alpha*lambda*sign(x(case2)))/(2*(alpha-1));
    
    y(case3) = (alpha + 1)*lambda^2/2;
    
end

