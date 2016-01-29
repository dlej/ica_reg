function [ S, W ] = ica_supergaussian_reg( X, Y, lambda, alpha )
%ICA_SUPERGAUSSIAN_REG ICA with sparse regression regularization
%   X and Y should be pre-whitened

% verbosity flag
verbose = false;

% contrast function an derivatives
G1 = @(x) log(cosh(x));
g1 = @(x) tanh(x);
dg1 = @(x) 1 - tanh(x).^2;

[p, n] = size(X);

converged = false;
r = 1e-2; % initial learning weight for bold driver

W = rand(p); % initialize W to a random orthogonal matrix
[W,~] = eig(W+W');

dW = 0;
f = 1e10; % put some ridiculous value here so we don't randomly guess
          % the next function value and think we've converged

for i=1:500 % terminate after 500 iterations
    W0 = W;
    dW0 = dW;
    f0 = f;
    fprintf('.');
    
    % update W
    W = W - r*dW;

    % re-orthogonalize W
    W = W/norm(W);
    while norm(W*W'-eye(p)) >= 1e-8
        W = 3/2*W - 1/2*W*W'*W;
    end
    
    % evaluate objective
    [f,dW] = ica_grad(W,X,Y,G1,g1,lambda,alpha);

    delta = (f - f0)/abs(f0);
    if delta < 1e-8 % gain more rate confidence if error goes down
        r = r*1.05;
        
    else
        r = r*0.5; % cut rate confidence if error goes up
        W = W0;  % undo the update
        dW = dW0;  % and eliminate momentum
        f = f0;
    end
    
    if verbose
        fprintf('Last cost delta: %.5f\n', delta);
    end
    
    if abs(delta) < 1e-5
        converged = true;
        break;
    end
end
fprintf('\n');

if ~converged
    fprintf('No convergence.\n');
end

S = W*X;

end

function [f, dW] = ica_grad( W, X, Y, G, g, lambda, alpha )

[p, ~] = size(X);

WX = W*X;
WY = W*Y;

f = 0;
dW = zeros(p);

for k = 1:p
    
    w = W(k,:)';
    wX = WX(k,:);
    wY = WY(k,:);
    
    gamma = mean(G(wX));
    
    [s, ds] = scad(wY, lambda, alpha);
    
    f = f + gamma + sum(s);

    dgamma = mean(bsxfun(@times, X, g(wX)), 2);
    
    Ys = Y*ds';
    dw = dgamma + Ys;
        
    dW(k,:) = dw';
end

end