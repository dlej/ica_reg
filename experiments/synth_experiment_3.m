% path to FastICA
addpath('../ext/FastICA_25')

% include our functions
addpath('../inc')

% initialize rng seed
rng(0);

n = 1e5 ; % number of samples [representative of edges in a component]
d = 30;  % number of components/patients
r = 10;   % number of phenotypes

Y_err = zeros(11,20,3);
noises = linspace(0,1,11);
for inoise = 1:11
for trial=1:20

% noise applied to various variables
noise_a = noises(inoise);
noise_x = 0.5;
noise_y = 0;

% generate mixture of gaussian and uniform, with the gaussian centered
% at 0 and with probability p, and the uniform on (-l, l). Gaussian
% variance is chosen so that the total variance is 1.
p = 0.98;
l = 6;
sigma = sqrt((1 - (1-p)*l^2/3)/p);

% let's also make the gaussian part have some correlation to an underlying
% signal. specify the correlation coefficient here
m = 50;
underlying = poissrnd(m^2/2, 1, n)/m;
% underlying = randn(1, n);


S = bsxfun(@minus, poissrnd(m^2/2, d, n), poissrnd(m^2/2, 1, n))/m;
% S = zeros(d,n);
% for j=1:d
%     rho = sign(rand - 0.5)*rand*0.2 + 0.5;
%     S(j,:) = underlying*rho + sqrt((1-rho^2))*randn(1,n);
% end
S(1:r,:) = sigma*S(1:r,:);
n_unif = floor(n*(1-p));
perm = randperm(n);
for j=1:r
    this_perm = perm(1:n_unif);
    perm = perm(n_unif+1:end);
    S(j,this_perm) = (rand(1,n_unif)-0.5)*2*l;
end



% generate random phenotypes
Y_true = randn(d,r);

% generate mixing matrix with some columns collinear with phenotypes
A_true = [Y_true(:,1:r) randn(d,d-r)];

% generate noisy observations of mixed signals
X = (A_true + randn(d)*noise_a)*S  + randn(d,n)*noise_x;

% noisily observe phenotypes
Y = Y_true + randn(d,r)*noise_y;

%%

% whiten X
X_mu = mean(X,2);
X_tilde = bsxfun(@minus,X,X_mu);
D = cov(X_tilde')^-(0.5);
X_tilde = D*X_tilde;

% run fastICA
[icasig, ~, W_fastica] = fastica(X_tilde, 'approach', 'symm', 'g', 'tanh');

% run our regularized ICA
% (these parameters seem reasonable for the time being)
lambda = 1;
alpha = 10;
[ S_reg, W_reg ] = ica_supergaussian_reg(X_tilde, W_fastica, D*Y, lambda, alpha);

%%

% compute regression coefficients for each solution
B1 = W_fastica*D*Y;
B2 = W_reg*D*Y;
B3 = zeros(d,r);
for j=1:r
    B3(:,j) = lasso(D\W_fastica',Y(:,j), 'Lambda', 1e-1);
end
% 
% % plot the regression coefficient matrices
% imagesc([abs(B1); zeros(1,r); 2*ones(1,r); zeros(1,r); abs(B2)], [0,1.2])
% colorbar

% now let's see how we do in regression with the top r components
% (selected in terms of highest sum of SCAD in each row of B
[~, I1] = sort(max(abs(B1),[],2), 'descend');
[~, I2] = sort(max(abs(B2),[],2), 'descend');
[~, I3] = sort(max(abs(B3),[],2), 'descend');

Y_err(inoise,trial,:) = [norm(Y_true - D\W_fastica(I1(1:r),:)'*W_fastica(I1(1:r),:)*D*Y, 'fro')/sqrt(d*r), ...
    norm(Y_true - D\W_fastica(I3(1:r),:)'*W_fastica(I3(1:r),:)*D*Y, 'fro')/sqrt(d*r), ...
    norm(Y_true - D\W_reg(I2(1:r),:)'*W_reg(I2(1:r),:)*D*Y, 'fro')/sqrt(d*r)];

end
end


%%
fastica_err = squeeze(Y_err(:,:,1));
fastica_lasso_err = squeeze(Y_err(:,:,2));
us_err = squeeze(Y_err(:,:,3));

errorbar(noises, mean(fastica_err,2), std(fastica_err,0,2));
hold on
errorbar(noises+0.01, mean(fastica_lasso_err,2), std(fastica_lasso_err,0,2));
errorbar(noises+0.02, mean(us_err,2), std(us_err,0,2));
hold off
legend('FastICA', 'FastICA + Lasso', 'Us')
title('Top-r Prediction Error')
xlabel('A noise')
ylabel('RMSE')