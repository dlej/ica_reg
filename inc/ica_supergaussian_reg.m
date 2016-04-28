function [ S, W ] = ica_supergaussian_reg( X, Y, D, lambda, alpha, penalty, nsplits, splititers )
%ICA_SUPERGAUSSIAN_REG ICA with sparse regression regularization
%   Input:
%       X           Data to be separated into (row) components. Should be
%                   pre-whitened.
%       Y           Target variables.
%       D           Matrix that was used to whiten X.
%       lambda      regularization parameter
%       alpha       SCAD parameter
%       penalty     'l1' or 'scad'
%       nsplits     multiple-start tournament depth
%       splititers  iterations between prunings in tournament
%
%   Output:
%       S           Recovered compnents
%       W           Orthogonal mixing matrix s.t. S=WX
%
% Gradient calculation is parallelized over features with parfor
% and will use matlabpool if available

% penalty
if(~exist('penalty','var'))
    alpha = 0;
    penalty = 0;
elseif strcmp(penalty, 'scad')
    penalty = 1;
else
    alpha = 0;
    penalty = 0;
end

% number of splits/branching factor
% we will start with 2^nsplits random mixing matrices
% and update them simultaneously, pruning half of the worst
% performers at regular intervals
if(~exist('nsplits','var'))
    nsimult = 1;
else
    nsimult = 2^nsplits;
end

% interations between splits
if(~exist('splititers','var'))
    splititers = 10;
end

% contrast function and derivatives
G1 = @(x) log(cosh(x));
g1 = @(x) tanh(x);
dg1 = @(x) 1 - tanh(x).^2;

[p, n] = size(X);

converged = false;
r = 1e-2*ones(nsimult,1); % initial learning weight for bold driver

% initialize W to random orthogonal matrices for the tournament
W = zeros(p,p,nsimult);
for k=1:nsimult
    W(:,:,k) = orthonormal_gen(p,p);
end

dW = zeros(p,p,nsimult);

% put some ridiculous value here so we don't randomly guess
% the next function value and think we've converged
f = 1e10*ones(nsimult,1); 

% start with all of the "players" in the "tournament"
keepers = 1:nsimult;

P = eye(p) - 1/p*ones(p);
Y_tilde = pinv(P/D)*(P*Y);
          
for i=1:500 % terminate after 500 iterations
    W0 = W;
    dW0 = dW;
    f0 = f;
    fprintf('.');
    
    % Do this for each of the "players" left in the "tournament"
    for k=keepers
        % update W
        W(:,:,k) = W(:,:,k) - r(k)*dW(:,:,k);
        
        % re-orthogonalize W
        W(:,:,k) = W(:,:,k)/norm(W(:,:,k));
        while norm(W(:,:,k)*W(:,:,k)'-eye(p)) >= 1e-8
            W(:,:,k) = 3/2*W(:,:,k) - 1/2*W(:,:,k)*W(:,:,k)'*W(:,:,k);
        end
    
        % evaluate objective
        [f_,dW_] = ica_grad(W(:,:,k),X,Y_tilde,G1,g1,lambda,alpha,penalty);
        f(k) = f_;
        dW(:,:,k) = dW_;

        delta = (f(k) - f0(k))/abs(f0(k));
        if delta < 1e-8 % gain more rate confidence if error goes down
            r(k) = r(k)*1.05;

        else
            r(k) = r(k)*0.5; % cut rate confidence if error goes up
            W(:,:,k) = W0(:,:,k);  % undo the update
            dW(:,:,k) = dW0(:,:,k);
            f(k) = f0(k);
        end
    end
    
    % Stop if we only have one "player" left and aren't changing
    if length(keepers) == 1 && abs(delta) < 1e-5
        converged = true;
        break;
    end
    
    % Prune the worst half of the players out
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
