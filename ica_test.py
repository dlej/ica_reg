import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import seaborn as sns

from ica_reg import ica_reg

# Generate mixed signals

p = 2

N = 10**4

s1 = np.sin((np.arange(N)+1)/200)
s2 = np.mod((np.arange(N)+1)/200, 2) - 1
S = np.concatenate([s.reshape((1,N)) for s in [s1, s2]], 0)
S = S - np.mean(S,1).reshape((p,1))

A = np.array([[1,2],[-2,1]])

X = A.dot(S)

#D = sp.linalg.sqrtm(np.linalg.inv(X.dot(X.T)/N))

icasig, W, B, b = ica_reg(X, np.zeros((2, 1)), alpha=0.001, lamda=0.0)

print(W, B, b)

plt.subplot(4,1,1)
plt.plot(X[0, :10000])
plt.subplot(4,1,2)
plt.plot(X[1, :10000])
plt.subplot(4,1,3)
plt.plot(icasig[0, :10000])
plt.subplot(4,1,4)
plt.plot(icasig[1, :10000])
plt.savefig('test.pdf')


