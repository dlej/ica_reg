% path to FastICA
addpath('../ext/FastICA_25')

% include our functions
addpath('../inc')

% initialize rng seed
rng(0);

n = 1e4; % number of samples [representative of edges in a component]
d = 10;  % number of components/patients
r = 5;   % number of phenotypes

% noise applied to various variables
noise_a = 0;
noise_x = 20;
noise_y = 0.0;

S = zeros(d,n);

% generate trimodal GMM with mass p near 0 and masses (1-p)/2 at +/-1
% choose variance for the conditional distribution at the peaks, 
% and necessary spacing will be chosen automatically such that the
% resulting distribution is unit variance.
p = 0.98;
sigma = 0.2;

for j=1:d
    l = sqrt((1-sigma^2)/(1-p)); % spacing
    for i=1:n
        if rand < p
            offset = 0;
        else
            offset = ((rand < 0.5)*2 - 1)*l;
        end
        S(j,i) = randn*sigma + offset;
    end
end

% generate random phenotypes
Y_true = randn(d,r);

% generate mixing matrix with some columns collinear with phenotypes
A_true = [Y_true(:,1:r) randn(d,d-r)];

% generate noisy observations of mixed signals
X = (A_true + (randn(d) - 0.5)*noise_a)*S  + (rand(d,n) - 0.5)*noise_x;

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
[ S_reg, W_reg ] = ica_supergaussian_reg(X_tilde, D*Y, lambda, alpha);

%%

% compute regression coefficients for each solution
B1 = W_fastica*D*Y;
B2 = W_reg*D*Y;

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
[~, I1] = sort(max(abs(B1),2), 'descend');
[~, I2] = sort(max(abs(B2),2), 'descend');

fprintf('\n')
fprintf('Top-r Regression error\n========\n')
fprintf('FastICA: %f, Us: %f\n', norm(Y_true - D\W_fastica(I1(1:r),:)'*W_fastica(I1(1:r),:)*D*Y, 'fro')/sqrt(d), ...
    norm(Y_true - D\W_reg(I2(1:r),:)'*W_reg(I2(1:r),:)*D*Y, 'fro')/sqrt(d))
