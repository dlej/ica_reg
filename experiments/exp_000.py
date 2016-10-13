#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.decomposition import FastICA
from sklearn.linear_model import LinearRegression

import sys
sys.path.append('..')
from ica_reg import ica_reg, tf_ica_supergaussian_obj

np.random.seed(42)

n = 10**5
d = 50
dd = 200
r = 15;
p = 0.98;
l = 6;
sigma = np.sqrt((1 - (1-p)*l**2/3)/p)

S = sigma*np.random.randn(dd, n)
unif = np.random.rand(dd, n) - p
S[unif > 0] = unif[unif > 0]*(2*l/(1-p)) - l;

Y_true = np.random.randn(d, r)
A_true = np.zeros((d, dd))
A_true[:, :r] = Y_true.copy()
A_true[:, r:] = np.random.randn(d, dd - r)

X = d/dd*A_true.dot(S)
X -= X.mean(1).reshape((d, 1))

fastica = FastICA()
S_fastica = fastica.fit_transform(X.T).T*np.sqrt(n)
A_fastica = fastica.mixing_/np.sqrt(n)
lr = LinearRegression()
B_fastica = lr.fit(A_fastica, Y_true).coef_.T

plt.figure()
plt.imshow(np.abs(B_fastica), interpolation="none", cmap='viridis')
#plt.clim(-2, 2)
plt.colorbar()
plt.savefig('exp_000a.pdf')

plt.figure()
plt.imshow(np.abs(S_fastica.dot(S.T)/n), interpolation="none", cmap='viridis', vmin=0, vmax=0.4)
plt.colorbar()
plt.savefig('exp_000aS.pdf')

S_hat, W_hat, B_hat = ica_reg(X, Y_true, alpha=1e-3, lamda=1e0, ica_obj=tf_ica_supergaussian_obj)

plt.figure()
plt.imshow(np.abs(B_hat), interpolation="none", cmap='viridis')
#plt.clim(-2, 2)
plt.colorbar()
plt.savefig('exp_000b.pdf')

plt.figure()
plt.imshow(np.abs(S_hat.dot(S.T)/n), interpolation="none", cmap='viridis', vmin=0, vmax=0.4)
plt.colorbar()
plt.savefig('exp_000bS.pdf')

