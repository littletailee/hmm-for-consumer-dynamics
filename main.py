#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import tensorflow as tf
import zhusuan as zs


def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1:
        p = p.flatten()

    return p


def hmm(observed, n_steps, n_particles):
    with zs.BayesianNet(observed=observed) as model:
        # p = zs.Uniform('p')
        # pi = tf.concat([p, tf.constant(1.)-p], axis=0)
        pi = np.array([0.5, 0.5]).astype(np.float32)
        q = zs.Normal('q', mean=[[0., 0.], [0., 0.]],
                      logstd=0., n_samples=n_particles, group_ndims=2)
        Q = tf.nn.softmax(q, axis=2)
        # Q = np.array([[0.9, 0.1], [0.1, 0.9]]).astype(np.float32)

        # p = zs.Normal('p', mean=[[0., 0.], [0., 0.]],
        #               logstd=0.)
        # P = tf.nn.softmax(p, axis=1)
        P = np.array([[0.9, 0.1], [0.1, 0.9]]).astype(np.float32)

        prob = tf.log(pi)
        prob = tf.tile(tf.expand_dims(prob, axis=0), [n_particles, 1])
        P = tf.log(P)
        Q = tf.log(Q)
        ta = tf.TensorArray(dtype=tf.float32, size=n_steps)

        # action_probs = []
        # for i in range(n_steps):
        #     action_prob = tf.reduce_logsumexp(
        #         tf.add(tf.expand_dims(P, axis=0),
        #                tf.expand_dims(prob, axis=2)),
        #         axis=1
        #     )
        #     action_probs.append(action_prob)
        #     prob = tf.reduce_logsumexp(
        #         tf.add(Q,
        #                tf.expand_dims(prob, axis=2)),
        #         axis=1
        #     )
        # action_prob = tf.stack(action_probs, axis=1)

        def loop_body(i, prob, ta):
            action_prob = tf.reduce_logsumexp(
                tf.add(tf.expand_dims(P, axis=0),
                       tf.expand_dims(prob, axis=2)),
                axis=1
            )
            ta = ta.write(i, action_prob)
            prob = tf.reduce_logsumexp(
                tf.add(Q,
                       tf.expand_dims(prob, axis=2)),
                axis=1
            )
            return i + 1, prob, ta

        i = tf.constant(0)
        i, prob, ta = tf.while_loop(lambda i, *args: tf.less(i, 10),
                                    loop_body, [i, prob, ta])

        action_prob = ta.stack()
        action_prob = tf.transpose(action_prob, [1, 0, 2])

        y = zs.Categorical('y', action_prob, group_ndims=1)

    return model


if __name__ == '__main__':
    tf.set_random_seed(42)
    T = 10
    y_observed = np.array([0] * T).astype(np.int32)

    n_chains = 100
    n_iters = 200
    burnin = n_iters // 2
    n_leapfrogs = 10

    def log_joint(observed):
        model = hmm(observed, T, n_chains)
        return model.local_log_prob('y')

    adapt_step_size = tf.placeholder(
        tf.bool, shape=[], name='adapt_step_size')
    adapt_mass = tf.placeholder(tf.bool, shape=[], name='adapt_mass')
    hmc = zs.HMC(step_size=1e-3, n_leapfrogs=n_leapfrogs,
                 adapt_step_size=adapt_step_size, adapt_mass=adapt_mass,
                 target_acceptance_rate=0.9)
    y = tf.placeholder(tf.int32, shape=[None])
    q = tf.Variable(tf.zeros([n_chains, 2, 2]), trainable=False, name='q')
    sample_op, hmc_info = hmc.sample(log_joint, {'y': y}, {'q': q})

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        samples = []
        print('Sampling...')
        for i in range(n_iters):
            _, p_sample, acc, ss = sess.run(
                [sample_op, hmc_info.samples['q'], hmc_info.acceptance_rate,
                 hmc_info.updated_step_size],
                feed_dict={adapt_step_size: i < burnin // 2,
                           adapt_mass: i < burnin // 2,
                           y: y_observed})
            if i % 10 == 0:
                print('Sample {}: Acceptance rate = {}, updated step size = {}'
                      .format(i, np.mean(acc), ss))
            if i >= burnin:
                samples.append(softmax(p_sample, axis=-1))
        print('Finished.')
        samples = np.vstack(samples)

    print('Sample mean = {}'.format(np.mean(samples, axis=0)))
    print('Sample stdev = {}'.format(np.std(samples, axis=0)))
