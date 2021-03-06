import numpy as np
import tensorflow as tf

from sklearn.decomposition import PCA

from bold_driver import BoldDriverOptimizer
from utils import gen_orthonormal


def tf_cosh(x):
    return (tf.exp(x) + tf.exp(-x)) / 2 

def tf_max_abs_row_sum_norm(A):
    return tf.reduce_max(tf.reduce_sum(tf.abs(A), reduction_indices=[1]))

def tf_l1_norm(A):
    return tf.reduce_sum(tf.abs(A))

def tf_l2_norm(A):
    return tf.sqrt(tf.reduce_sum(tf.pow(A, 2)))

def tf_l1_l2_ratio(A):
    n = np.prod([d.value for d in A.get_shape()])
    return (tf_l1_norm(A) / tf_l2_norm(A) - 1) / (n**0.5 - 1)

def tf_orthogonalize(W, eps=1e-6, back_prop=False):
    p = W.get_shape()[0].value
    eye = tf.constant(np.eye(p, dtype=np.float32))
    def ortho_step(Q0):
        Q1 = Q0 / tf.sqrt(tf_max_abs_row_sum_norm(tf.matmul(Q0, Q0, transpose_b=True)))
        return 3/2*Q1 - 0.5*tf.matmul(Q1, tf.matmul(Q1, Q1, transpose_a=True))
    c = lambda Q: tf.greater(tf_max_abs_row_sum_norm(tf.matmul(Q, Q, transpose_a=True) - eye)/p, eps)
    b = lambda Q: ortho_step(Q)
    return tf.while_loop(c, b, [W], back_prop=back_prop)

def tf_ica_obj(W, X):
    G = tf.log(tf_cosh(tf.matmul(W, X)))
    EG = tf.reduce_mean(G, reduction_indices=[1])
    nu = np.log(np.cosh(np.random.randn(10**6))).mean()
    return -tf.reduce_mean(tf.pow(EG - nu*tf.ones_like(EG), 2))

def tf_ica_supergaussian_obj(W, X):
    G = tf.log(tf_cosh(tf.matmul(W, X)))
    return tf.reduce_mean(tf.reduce_mean(G, reduction_indices=[1]))

def tf_ica_subgaussian_obj(W, X):
    return -tf_ica_supergaussian_obj(W, X)

def ica_reg(X, Y, alpha=1.0, lamda=1.0, ica_obj=tf_ica_obj):
    p, n = X.shape
    p_, r = Y.shape
    if p != p_:
        raise ValueError('X and Y must have the same number of rows.')

    pca = PCA(whiten=True)
    X_w = pca.fit_transform(X.T).T
    D = pca.components_
    P = np.eye(p) - np.ones(p)/p
    Y_tilde = tf.constant(np.linalg.pinv(P.dot(np.linalg.pinv(D))).dot(P.dot(Y)), dtype=tf.float32)
    #D_inv = tf.constant(np.linalg.pinv(pca.components_), dtype=tf.float32)

    W = tf.placeholder(tf.float32, shape=[p, p])
    #B = tf.placeholder(tf.float32, shape=[p, r])
    #b = tf.placeholder(tf.float32, shape=[r])
    B = tf.matmul(W, Y_tilde)
    B_err = tf.placeholder(tf.float32, shape=[p, r])

    W_ortho = tf_orthogonalize(W)
    B_err_null = tf.matmul(B, tf.matrix_solve_ls(B, B_err))

    #A = tf.matmul(D_inv, tf.transpose(W))
    #Y_hat = tf.matmul(A, B) + tf.tile(tf.reshape(b, [1, r]), [p, 1])
    
    J_ica = ica_obj(W, tf.constant(X_w, dtype=tf.float32)) 
    #J_regression = 0.5*tf_l2_norm(Y - Y_hat)
    J_sparse = tf_l1_norm(B + B_err)

    #J = J_ica + alpha*J_regression + lamda*J_sparse
    J = J_ica + lamda*J_sparse

    with tf.Session() as sess:
        bd_opt = BoldDriverOptimizer(sess, J, [W, B_err], \
                [gen_orthonormal(p), np.zeros_like(Y)], \
                [W_ortho, B_err_null])
        while not bd_opt.converged:
            bd_opt.run()
            print(bd_opt.f)
        _W, _B_err = bd_opt.x
        feed_dict = {W: _W, B_err: _B_err}
        _B = (B + B_err).eval(feed_dict=feed_dict)
        print(sess.run([J_ica, J_sparse], feed_dict=feed_dict))

    S = _W.dot(X_w)

    return S, _W, _B
    

