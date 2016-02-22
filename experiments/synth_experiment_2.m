% path to FastICA
addpath('../ext/FastICA_25')

% include our functions
addpath('../inc')

% initialize rng seed
rng(0);

n = 1e5; % number of samples [representative of edges in a component]
d = 200;  % number of patients
dd = 1000;  % number of components
r = 15;   % number of phenotypes

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


% S = bsxfun(@minus, poissrnd(m^2/2, d, n), poissrnd(m^2/2, 1, n))/m;
% S = zeros(d,n);
% for j=1:d
%     rho = sign(rand - 0.5)*rand*0.2 + 0.5;
%     S(j,:) = underlying*rho + sqrt((1-rho^2))*randn(1,n);
% end
% S(1:r,:) = sigma*S(1:r,:);
S = sigma*randn(dd, n);
unif = (rand(dd, n) - p);
S(unif > 0) = unif((unif > 0))*(2*l/(1-p)) - l;
% n_unif = floor(n*(1-p));
% perm = randperm(n);
% for j=1:r
%     this_perm = perm(1:n_unif);
%     perm = perm(n_unif+1:end);
%     S(j,this_perm) = (rand(1,n_unif)-0.5)*2*l;
% end

clear unif;

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

% run fastICA
[icasig, ~, W_fastica] = fastica(X_tilde, 'approach', 'symm', 'g', 'tanh');

% run our regularized ICA
% (these parameters seem reasonable for the time being)
lambda = 1;
alpha = 2;
[ S_reg, W_reg ] = ica_supergaussian_reg(X_tilde, D*Y, lambda, alpha, true);

%%

% compute regression coefficients for each solution
B1 = W_fastica*D*Y;
B2 = W_reg*D*Y;
B3 = zeros(d,r);
for j=1:r
    B3(:,j) = lasso(D\W_fastica',Y(:,j), 'Lambda', 1e-1);
end

% plot the regression coefficient matrices
imagesc([abs(B1); zeros(1,r); 2*ones(1,r); zeros(1,r); abs(B2)], [0,1.2])
colorbar

% here's the SCAD penalty of each regression coefficient matrix
% just to show that we are indeed minimizing this
B1scad = scad(B1, lambda, alpha);
B2scad = scad(B2, lambda, alpha); 
fprintf('SCAD\n========\nFastICA: %f, Us: %f\n', sum(B1scad(:)), sum(B2scad(:)))

% now let's see how we do in regression with the top r components
% (selected in terms of highest sum of SCAD in each row of B
[~, I1] = sort(max(abs(B1),[],2), 'descend');
[~, I2] = sort(max(abs(B2),[],2), 'descend');
[~, I3] = sort(max(abs(B3),[],2), 'descend');

fprintf('\n')
rr = floor(r*5);
fprintf('Top-r Regression error\n========\n')
fprintf('FastICA: %f, FastICA+Lasso: %f, Us: %f\n', norm(Y_true - D\W_fastica(I1(1:rr),:)'*W_fastica(I1(1:rr),:)*D*Y, 'fro')/sqrt(d*r), ...
    norm(Y_true - D\W_fastica(I3(1:rr),:)'*W_fastica(I3(1:rr),:)*D*Y, 'fro')/sqrt(d*r), ...
    norm(Y_true - D\W_reg(I2(1:rr),:)'*W_reg(I2(1:rr),:)*D*Y, 'fro')/sqrt(d*r))

%%
C1 = (icasig).*(abs(icasig) > 2)*((S(1:r,:)).*(abs(S(1:r,:)) > 3))'/mean(sum(abs(S(1:r,:)) > 3, 2));
C2 = (S_reg).*(abs(S_reg) > 2)*((S(1:r,:)).*(abs(S(1:r,:)) > 3))'/mean(sum(abs(S(1:r,:)) > 3, 2));
% C1 = (icasig)*((S(1:r,:)).*(abs(S(1:r,:)) > 3))'/mean(sum(abs(S(1:r,:)) > 3, 2));
% C2 = (S_reg)*((S(1:r,:)).*(abs(S(1:r,:)) > 3))'/mean(sum(abs(S(1:r,:)) > 3, 2));

imagesc([C1; [-10;10;-10]*ones(1,r); C2], [-1,1]*3)
colorbar
title('Extremity correlation for FastICA(top) and Us(bottom), n=200, s/n=5')