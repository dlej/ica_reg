% path to FastICA
addpath('../ext/FastICA_25')

% include our functions
addpath('../inc')

% initialize rng seed
rng(0);

ntrials = 20;
nalpha = 3;
nlambda = 21;

Y_err = zeros(nlambda,nalpha+1,ntrials);
lambdas = logspace(-3,3,nlambda);
alphas = linspace(1,nalpha,nalpha);

for ilambda=1:nlambda
lambda = lambdas(ilambda);

for itrial=1:ntrials
fprintf('%d,%d', ilambda, itrial);

n = 1e4; % number of samples [representative of edges in a component]
d = 20;  % number of patients
dd = 100;% number of components
r = 5;   % number of phenotypes
rr = 5;

% noise applied to various variables
noise_a = 0;
noise_x = 0;
noise_y = 0;

% generate mixture of gaussian and uniform, with the gaussian centered
% at 0 and with probability p, and the uniform on (-l, l). Gaussian
% variance is chosen so that the total variance is 1.
p = 0.98;
l = 6;
sigma = sqrt((1 - (1-p)*l^2/3)/p);

% let's also make the gaussian part have some correlation to an underlying
% signal. specify the correlation coefficient here
% m = 50;
% underlying = poissrnd(m^2/2, 1, n)/m;
% underlying = randn(1, n);

S = sigma*randn(dd, n);
unif = (rand(dd, n) - p);
S(unif > 0) = unif((unif > 0))*(2*l/(1-p)) - l;

% generate random phenotypes
Y_true = randn(d,r);

% generate mixing matrix with some columns collinear with phenotypes
A_true = [Y_true(:,1:r) randn(d,dd-r)];

% generate noisy observations of mixed signals
X = (A_true + randn(d,dd)*noise_a)*d/dd*S  + randn(d,n)*noise_x;

% noisily observe phenotypes
Y = Y_true + randn(d,r)*noise_y;

%%

% whiten X
X_mu = mean(X,2);
X_tilde = bsxfun(@minus,X,X_mu);
D = cov(X_tilde')^-(0.5);
X_tilde = D*X_tilde;

% run our regularized ICA
[ S_reg, W_reg ] = ica_supergaussian_reg(X_tilde, D*Y, lambda, 0, false);
B = W_reg*D*Y;
[~, I] = sort(sum(B.^2,2), 'descend');
Y_err(ilambda,1,itrial) = ...
        norm(Y_true - D\W_reg(I(1:rr),:)'*W_reg(I(1:rr),:)*D*Y, 'fro')/sqrt(d*r);

for ialpha = 1:nalpha
    alpha = alphas(ialpha);
    [ S_reg, W_reg ] = ica_supergaussian_reg(X_tilde, D*Y, lambda, alpha, false, 'scad');
    B = W_reg*D*Y;
    [~, I] = sort(sum(B.^2,2), 'descend');
    Y_err(ilambda,ialpha+1,itrial) = ...
        norm(Y_true - D\W_reg(I(1:rr),:)'*W_reg(I(1:rr),:)*D*Y, 'fro')/sqrt(d*r);
end


end
end

%%
% yerr1 = quantile(squeeze(Y_err(:,1,:)),0.25,2);
yerr2 = mean(squeeze(Y_err(:,1,:)),2);
% yerr3 = quantile(squeeze(Y_err(:,1,:)),0.75,2);
semilogx(lambdas, yerr2);
ax = gca;
hold on
% ax.ColorOrderIndex = 1;
% semilogx(lambdas, yerr1, '--');
% ax.ColorOrderIndex = 1;
% semilogx(lambdas, yerr3, '--');

for ialpha = 1:nalpha
    yerr2 = mean(squeeze(Y_err(:,1+ialpha,:)),2);
%     yerr1 = quantile(squeeze(Y_err(:,1+ialpha,:)),0.25,2);
%     yerr3 = quantile(squeeze(Y_err(:,1+ialpha,:)),0.75,2);
    ax.ColorOrderIndex = ialpha+1;
    semilogx(lambdas, yerr2);
%     ax.ColorOrderIndex = ialpha+1;
%     semilogx(lambdas, yerr1, '--');
%     ax.ColorOrderIndex = ialpha+1;
%     semilogx(lambdas, yerr3, '--');
end

hold off
title('Top-rows (L2) Prediction Error, s/n=3, rows=33')
legend('L1', 'SCAD, \alpha=1', 'SCAD, \alpha=2', 'SCAD, \alpha=3')
xlabel('\lambda')
ylabel('RMSE')