import tensorflow as tf
import numpy as np
import unittest

class BoldDriverOptimizer():

    def __init__(self, sess, f, x, x0, proj=None, r=1.0, rho=1.2, 
            sigma=0.5, rtol=1e-5, atol=1e-8):
        self._sess = sess
        self._tf_f = f
        self._tf_x = x
        self._tf_dx = tf.gradients([f], x)
        self.x = x0
        self.f = 1e100
        self._f_next = None
        if proj is None:
            self._tf_proj = [None for _ in x]
        else:
            self._tf_proj = proj
        self._r = r
        self._rho = rho
        self._sigma = sigma
        self._rtol = rtol
        self._atol = atol
        self.iters = 0
        self.converged = False

    def run(self):
        f0 = self.f
        x0 = self.x[:]
        
        feed_dict0 = {tf_x: x for tf_x, x in zip(self._tf_x, x0)}
        res = self._sess.run([self._tf_f] + self._tf_dx, feed_dict=feed_dict0)
        self.f = res[0]
        dx0 = res[1:]

        self.x = [x - self._r*dx for x, dx in zip(x0, dx0)]
        self.x = [x if proj is None else proj.eval(feed_dict={tf_x: x}) \
                for x, tf_x, proj in zip(self.x, self._tf_x, self._tf_proj)]

        feed_dict = {tf_x: x for tf_x, x in zip(self._tf_x, self.x)}
        self._f_next = self._tf_f.eval(feed_dict=feed_dict)
        if self._f_next > self.f:
            self.x = x0
            self.f = f0
            self._r *= self._sigma
            self.converged = False
        else:
            self._r *= self._rho
            self.converged = np.allclose(self.f, self._f_next, rtol=self._rtol, 
                    atol=self._atol) and \
                    all(np.allclose(a, b, rtol=self._rtol, atol=self._atol) 
                            for a, b in zip(self.x, x0))
        self.iters += 1


class TestBoldDriver(unittest.TestCase):

    def test_one_var(self):
        x = tf.placeholder(tf.float32, shape=[])
        f = tf.pow(x - 2, 2)
        with tf.Session() as sess:
            bdopt = BoldDriverOptimizer(sess, f, [x], [0])
            while not bdopt.converged:
                bdopt.run()
        self.assertTrue(np.allclose(bdopt.x, [2]))
        self.assertTrue(np.allclose(bdopt.f, 0))

    def test_two_vars_proj(self):
        x = tf.placeholder(tf.float32, shape=[2])
        y = tf.placeholder(tf.float32, shape=[2])
        f = tf.reduce_sum(tf.pow(x - y, 2)) + tf.reduce_sum(tf.pow(x - 2, 2))
        x_bounded = tf.clip_by_value(x, -1, 1)
        with tf.Session() as sess:
            bdopt = BoldDriverOptimizer(sess, f, [x, y], [np.zeros(2), 
                np.zeros(2)], [x_bounded, None])
            while not bdopt.converged:
                bdopt.run()
        self.assertTrue(np.allclose(bdopt.x[0], [1, 1]))
        self.assertTrue(np.allclose(bdopt.x[1], [1, 1]))


if __name__ == '__main__':
    unittest.main()



