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


def hmm(observed, n_steps):
    with zs.BayesianNet(observed=observed) as model:
        p = zs.Uniform('p')
        pi = tf.concat([p, tf.constant(1.)-p], axis=0)
        Q = np.array([[0.9, 0.1], [0.1, 0.9]]).astype(np.float32)
        P = np.array([[0.9, 0.1], [0.1, 0.9]]).astype(np.float32)

        prob = tf.log(pi)
        P = tf.log(P)
        Q = tf.log(Q)
        actions_probs = []

        for i in range(n_steps):
            action_prob = tf.reduce_logsumexp(
                tf.add(P, tf.expand_dims(prob, axis=0)),
                axis=1
            )
            actions_probs.append(action_prob)
            prob = tf.reduce_logsumexp(
                tf.add(Q, tf.expand_dims(prob, axis=0)),
                axis=1
            )

        action_prob = tf.stack(actions_probs, axis=0)
        y = zs.Categorical('y', action_prob, group_ndims=1)

    return model


if __name__ == '__main__':
    tf.set_random_seed(42)
    T = 10
    y_observed = np.array([0] * T).astype(np.int32)

    n_chains = 1
    n_iters = 200
    burnin = n_iters // 2
    n_leapfrogs = 10

    # model = hmm({'y': y}, T)

    def log_joint(observed):
        model = hmm(observed, T)
        return model.local_log_prob('y')

    adapt_step_size = tf.placeholder(
        tf.bool, shape=[], name='adapt_step_size')
    adapt_mass = tf.placeholder(tf.bool, shape=[], name='adapt_mass')
    hmc = zs.HMC(step_size=1e-3, n_leapfrogs=n_leapfrogs,
                 adapt_step_size=adapt_step_size, adapt_mass=adapt_mass,
                 target_acceptance_rate=0.9)
    y = tf.placeholder(tf.int32, shape=[None])
    p = tf.Variable(tf.zeros([n_chains]) + 0.5, trainable=False, name='p')
    sample_op, hmc_info = hmc.sample(log_joint, {'y': y}, {'p': p})

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        samples = []
        print('Sampling...')
        for i in range(n_iters):
            _, p_sample, acc, ss = sess.run(
                [sample_op, hmc_info.samples['p'], hmc_info.acceptance_rate,
                 hmc_info.updated_step_size],
                feed_dict={adapt_step_size: i < burnin // 2,
                           adapt_mass: i < burnin // 2,
                           y: y_observed})
            print('Sample {}: Acceptance rate = {}, updated step size = {},'
                  .format(i, np.mean(acc), ss))
            if i >= burnin:
                samples.append(p_sample)
        print('Finished.')
        samples = np.vstack(samples)

    print('Sample mean = {}'.format(np.mean(samples)))
    print('Sample stdev = {}'.format(np.std(samples)))
