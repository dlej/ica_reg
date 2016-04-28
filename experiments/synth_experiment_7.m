% include our functions
addpath('../inc')

load('../../samplestats.mat');

%%
% initialize rng seed
rng(0);

C = stats.x.rowvar;
[U,Lambda] = eig(C);

ntrials = 1;
nalpha = 1;
nlambda = 1;
nsplits = 2;

Y_err = zeros(nlambda,nalpha+1,ntrials);
Ws = zeros(519,519,nlambda);
lambdas = [1];
alphas = [2];

for itrial=1:ntrials


n = 5e5; % number of samples [representative of edges in a component]
d = 519;  % number of patients
dd = 5000;% number of components
r = 1;   % number of phenotypes
rr = 15;

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

% generate mixing matrix with some columns collinear with phenotypes
A_true = U*diag(sqrt(diag(Lambda)))*orthonormal_gen(dd,d)';

Y_true = randn(d, r);

% generate noisy observations of mixed signals
X = A_true*S;

% noisily observe phenotypes
Y = Y_true;

%%

% whiten X
X_mu = mean(X,2);
X_tilde = bsxfun(@minus,X,X_mu);
D = cov(X_tilde')^-(0.5);
X_tilde = D*X_tilde;

% addpath('../../FastICA_25');
% [icasig, ~, W_fastica] = fastica(X_tilde, 'approach', 'symm', 'g', 'tanh');
% B_fica = W_fastica*D*Y;
% [~, I] = sort(sum(B_fica.^2,2), 'descend');
% Y_err_fica = ...
%         norm(Y_true - D\W_fastica(I(1:rr),:)'*W_fastica(I(1:rr),:)*D*Y, 'fro')/sqrt(d*r);
% B3 = zeros(d,r);
% for j=1:r
%     B3(:,j) = lasso(D\W_fastica',Y(:,j), 'Lambda', 1e-2);
% end
% [~, I] = sort(sum(B3.^2,2), 'descend');
% Y_err_fica_lasso = ...
%         norm(Y_true - D\W_fastica(I(1:rr),:)'*W_fastica(I(1:rr),:)*D*Y, 'fro')/sqrt(d*r);
    
for ilambda=1:nlambda
fprintf('%d,%d', ilambda, itrial);
lambda = lambdas(ilambda);

% run our regularized ICA
[ S_reg, W_reg ] = ica_supergaussian_reg(X_tilde, D*Y, lambda, 0, false, 'l1', nsplits);
Ws(:,:,ilambda) = W_reg;
% W_reg = Ws(:,:,ilambda);

B = W_reg*D*Y;
[~, I] = sort(sum(B.^2,2), 'descend');
Y_err(ilambda,1,itrial) = ...
        norm(Y_true - D\W_reg(I(1:rr),:)'*W_reg(I(1:rr),:)*D*Y, 'fro')/sqrt(d*r);

fprintf('%.5f\n', Y_err(ilambda,1,itrial));
fprintf('%.5f\n', sum(abs(B(:))));

% for ialpha = 1:nalpha
%     alpha = alphas(ialpha);
%     [ S_reg, W_reg ] = ica_supergaussian_reg(X_tilde, D*Y, lambda, alpha, false, 'scad', nsplits);
%     B = W_reg*D*Y;
%     [~, I] = sort(sum(B.^2,2), 'descend');
%     Y_err(ilambda,ialpha+1,itrial) = ...
%         norm(Y_true - D\W_reg(I(1:rr),:)'*W_reg(I(1:rr),:)*D*Y, 'fro')/sqrt(d*r);
% end


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

% for ialpha = 1:nalpha
%     yerr2 = mean(squeeze(Y_err(:,1+ialpha,:)),2);
% %     yerr1 = quantile(squeeze(Y_err(:,1+ialpha,:)),0.25,2);
% %     yerr3 = quantile(squeeze(Y_err(:,1+ialpha,:)),0.75,2);
%     ax.ColorOrderIndex = ialpha+1;
%     semilogx(lambdas, yerr2);
% %     ax.ColorOrderIndex = ialpha+1;
% %     semilogx(lambdas, yerr1, '--');
% %     ax.ColorOrderIndex = ialpha+1;
% %     semilogx(lambdas, yerr3, '--');
% end

plot([min(lambdas), max(lambdas)], Y_err_fica*[1, 1])

hold off
title('Top-rows (L2) Prediction Error, s/n=2, rows=30, #phenotypes=5')
% legend('L1', 'SCAD, \alpha=1', 'SCAD, \alpha=2', 'SCAD, \alpha=3')
legend('L1 + our method', 'FastICA')
xlabel('\lambda')
ylabel('RMSE')