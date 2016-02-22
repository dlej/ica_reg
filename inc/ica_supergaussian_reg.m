function [ S, W ] = ica_supergaussian_reg( X, Y, lambda, alpha, verbose, penalty, splits, nsplititers )
%ICA_SUPERGAUSSIAN_REG ICA with sparse regression regularization
%   X and Y should be pre-whitened
% Gradient calculation is parallelized over features with parfor
% and will use matlabpool if available

% verbosity flag
if(~exist('verbose','var'))
    verbose = false;
end

% penalty
if(~exist('penalty','var'))
    penalty = 0;
elseif strcmp(penalty, 'scad')
    penalty = 1;
else
    penalty = 0;
end

% number of splits
if(~exist('nsplits','var'))
    nsimult = 1;
else
    nsimult = 2^nsplits;
end

% interations between splits
if(~exist('splititers','var'))
    splititers = 10;
end

% contrast function an derivatives
G1 = @(x) log(cosh(x));
g1 = @(x) tanh(x);
dg1 = @(x) 1 - tanh(x).^2;

[p, n] = size(X);

converged = false;
r = 1e-2*ones(nsimult); % initial learning weight for bold driver

% initialize W to a random orthogonal matrices
W = zeros(p,p,nsimult);
for k=1:nsimult
    Wgen = randn(p); 
    [Wgen,~] = eig(Wgen*Wgen');
    W(:,:,k) = Wgen;
end

dW = zeros(p,p,nsimult);
f = 1e10*ones(nsimult); % put some ridiculous value here so we don't randomly guess
          % the next function value and think we've converged

keepers = 1:nsimult;
          
for i=1:500 % terminate after 500 iterations
    W0 = W;
    dW0 = dW;
    f0 = f;
    fprintf('.');

    % re-orthogonalize W
    for k=keepers
        % update W
        W(:,:,k) = W(:,:,k) - r(k)*dW(:,:,k);

        W(:,:,k) = W(:,:,k)/norm(W(:,:,k));
        while norm(W(:,:,k)*W(:,:,k)'-eye(p)) >= 1e-8
            W(:,:,k) = 3/2*W(:,:,k) - 1/2*W(:,:,k)*W(:,:,k)'*W(:,:,k);
        end
    
        % evaluate objective
        [f_,dW_] = ica_grad(W(:,:,k),X,Y,G1,g1,lambda,alpha,penalty);
        f(k) = f_;
        dW(:,:,k) = dW_;

        delta = (f(k) - f0(k))/abs(f0(k));
        if delta < 1e-8 % gain more rate confidence if error goes down
            r(k) = r(k)*1.05;

        else
            r(k) = r(k)*0.5; % cut rate confidence if error goes up
            W(:,:,k) = W0(:,:,k);  % undo the update
            dW(:,:,k) = dW0(:,:,k);  % and eliminate momentum
            f(k) = f0(k);
        end
    end
    
    if verbose
        fprintf('Last cost delta: %.5f\n', delta);
    end
    
    if length(keepers) == 1 && abs(delta) < 1e-5
        converged = true;
        break;
    end
    
    kk = length(keepers);
    if kk > 1 && mod(i,splititers) == 0
        [~,I] = sort(f);
        kk = kk/2;
        keepers = I(1:kk);
    end
    
end
fprintf('\n');

if ~converged
    fprintf('No convergence.\n');
end

W = squeeze(W(:,:,keepers));
S = W*X;

end

function [f, dW] = ica_grad( W, X, Y, G, g, lambda, alpha, penalty )

[p, ~] = size(X);

WX = W*X;
WY = W*Y;

f = 0;
dW = zeros(p);

parfor k = 1:p
    
    w = W(k,:)';
    wX = WX(k,:);
    wY = WY(k,:);
    
    gamma = mean(G(wX));
    
    if penalty == 1
        [s, ds] = scad(wY, lambda, alpha);
    else
        s = lambda*abs(wY);
        ds = lambda*sign(wY);
    end
    
    f = f + gamma + sum(s);

    dgamma = mean(bsxfun(@times, X, g(wX)), 2);
    
    Ys = Y*ds';
    dw = dgamma + Ys;
        
    dW(k,:) = dw';
end

end
