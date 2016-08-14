import tensorflow as tf
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from bold_driver import BoldDriverOptimizer
import seaborn as sns

def tf_cosh(x):
    return (tf.exp(x) + tf.exp(-x)) / 2 

def tf_max_abs_row_sum_norm(A):
    return tf.reduce_max(tf.reduce_sum(tf.abs(A), reduction_indices=[1]))

def tf_orthogonalize(W, eps=1e-6, back_prop=False):
    p = W.get_shape()[0].value
    eye = tf.constant(np.eye(p, dtype=np.float32))
    def ortho_step(Q0):
        Q1 = Q0 / tf.sqrt(tf_max_abs_row_sum_norm(tf.matmul(Q0, Q0, transpose_b=True)))
        return 3/2*Q1 - 0.5*tf.matmul(Q1, tf.matmul(Q1, Q1, transpose_a=True))
    c = lambda Q: tf.greater(tf_max_abs_row_sum_norm(tf.matmul(Q, Q, transpose_a=True) - eye)/p, eps)
    b = lambda Q: ortho_step(Q)
    return tf.while_loop(c, b, [W], back_prop=back_prop)

def tf_ica_obj(W, x):
    G = tf.log(tf_cosh(tf.matmul(W, x)))
    EG = tf.reduce_mean(G, reduction_indices=[1])
    nu = np.log(np.cosh(np.random.randn(10**6))).mean()
    return -tf.reduce_sum(tf.pow(EG - nu*tf.ones_like(EG), 2))

def tf_ica_supergaussian_obj(W, x):
    G = tf.log(tf_cosh(tf.matmul(W, x)))
    return tf.reduce_sum(tf.reduce_mean(G, reduction_indices=[0]))

def tf_ica_subgaussian_obj(W, x):
    return -tf_ica_supergaussian_obj(W, x)


# Generate mixed signals

p = 2

N = 10**4

s1 = np.sin((np.arange(N)+1)/200)
s2 = np.mod((np.arange(N)+1)/200, 2) - 1
S = np.concatenate([s.reshape((1,N)) for s in [s1, s2]], 0)
S = S - np.mean(S,1).reshape((p,1))

A = np.array([[1,2],[-2,1]])

X = A.dot(S)

D = sp.linalg.sqrtm(np.linalg.inv(X.dot(X.T)/N))

X_w = D.dot(X)

# Create Tensorflow graph

#x = tf.placeholder(tf.float32, shape=[None, p])
x = tf.constant(X_w, dtype=tf.float32)
W = tf.placeholder(tf.float32, shape=[p, p])
W_ortho = tf_orthogonalize(W)
J = tf_ica_subgaussian_obj(W, x)

with tf.Session() as sess:
    W0 = W_ortho.eval(feed_dict={W: np.random.randn(p, p)})
    bdopt = BoldDriverOptimizer(J, [W], [W0], [W_ortho])

    while not bdopt.converged:
        bdopt.run()
        print(bdopt.f)
    print(bdopt.iters)
    print(bdopt.x[0])


icasig = bdopt.x[0].dot(X_w)

plt.subplot(4,1,1)
plt.plot(X[0, :10000])
plt.subplot(4,1,2)
plt.plot(X[1, :10000])
plt.subplot(4,1,3)
plt.plot(icasig[0, :10000])
plt.subplot(4,1,4)
plt.plot(icasig[1, :10000])
plt.savefig('test.pdf')


