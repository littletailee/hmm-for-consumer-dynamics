import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import zhusuan as zs


X_DIM = 1
A_DIM = 2
Z_DIM = 3


def read_data():
    dat = pd.read_csv('act2000b.dat', sep='	')
    dat['volunteer'] = dat['volunteer'].apply(lambda x: 0 if x == '.' else 1)
    int_loc = [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15]
    dat.iloc[:, int_loc] = dat.iloc[:, int_loc].apply(
        lambda x: x.apply(int, 1))
    dat.iloc[:, [2, 3]] = dat.iloc[:, [2, 3]].apply(
        lambda x: x.apply(float, 1))
    dat['time_after_grad'] = dat['Year'] - dat['grad year']
    dat['reunion_year'] = np.zeros(len(dat.iloc[:, 1]))
    row = dat[dat['reuniun'] == 1].index
    dat['reunion_year'][row] = dat['Year'][row]

    grouped = dat.groupby('ID')
    c = grouped[['Year']].agg(['count'])

    IDs = c[c.iloc[:, 0] == 23].index
    df = dat[dat['ID'].isin(IDs)]
    names = list(df)
    sequences = df.groupby(['ID'])
    data = sequences[names].apply(lambda x: x.values.tolist()).tolist()

    del grouped, c, df, sequences, IDs, int_loc, row

    # ----------------------------------here we go-----------------------------------------------------------

    # all data in tensor_all: array(#ID 741* #year 23* #variable 17)
    # variables: 0'ID', 1'Year',2'amount', 3'lag_amount', 4'Gift (yes/no)', 5'Lag_gift', 6'Years in data',
    #           7'Gift years', 8'SAA member',  9'Spouse alum', 10'number of degrees', 11'grad year', 12'reuniun',
    #           13'award', 14'volunteer', 15'Event', 16'time_after_grad'，17'reunion_year'
    tensor_all = np.array(data)

    # a_it: array(741, 23, 3)
    # variables: 0'reuniun', 1'volunteer', 2'Year'
    a_it = tensor_all[:, :, [12, 14, 1]]

    # x_it: array(741, 23, 3)
    # variables: 0'reunion year', 1'time after graduation'
    # fail to find reunion year-------------------
    x_it = tensor_all[:, :, [16]]
    # potential one:
    # x_it = tensor_all[:,:,[16,17]]

    # y_it: array(741, 23, 1)
    # variable: 0'Gift(yes/no)'
    y_it = tensor_all[:, :, 4]

    # z_it: array(741, 23, 3) (personal information, static)
    # variable: 0'SAA member', 1'Spouse alum', 2'number of degrees'
    z_it = tensor_all[:, 0, [8, 9, 10]]

    return a_it, x_it, y_it, z_it


def hmm(observed, x, a, z, n_batch, n_steps, n_particles):
    with zs.BayesianNet(observed=observed) as model:
        beta = zs.Normal('beta', mean=[0., 0., 0.],
                         std=10., n_samples=n_particles, group_ndims=1)
        beta0 = zs.Normal('beta0', mean=[0., 0., 0.],
                          std=10., n_samples=n_particles, group_ndims=1)
        delta = zs.Normal('delta', mean=[[[0.] * Z_DIM] * 2] * 3, std=10.,
                          n_samples=n_particles, group_ndims=3)
        epsilon = zs.Normal('epsilon', mean=[[[0.] * 3] * 3] * n_batch, std=0.01,
                            n_samples=n_particles, group_ndims=3)
        rho = zs.Normal('rho', mean=[[0.] * A_DIM] * 3, std=10.,
                        n_samples=n_particles, group_ndims=3)

        beta0 = tf.stack([beta0[:, 0],
                          beta0[:, 0] + tf.exp(beta0[:, 1]),
                          beta0[:, 0] + tf.exp(beta0[:, 1]) + tf.exp(beta0[:, 2])], axis=1)

        pi = np.array([1. / 3., 1. / 3, 1. / 3]).astype(np.float32)

        # [np, nb, 3, 3]
        mu = tf.reduce_sum(tf.multiply(
            tf.expand_dims(delta, axis=1),
            z[tf.newaxis, :, tf.newaxis, tf.newaxis, :]
        ), axis=4)

        prob = tf.tile(tf.expand_dims(pi, axis=0), [n_batch, 1])
        prob = tf.tile(tf.expand_dims(prob, axis=0), [n_particles, 1, 1])
        ta = tf.TensorArray(dtype=tf.float32, size=n_steps)

        def loop_body(i, prob, ta):
            logits = tf.reduce_sum(beta[:, tf.newaxis, :, tf.newaxis] *
                                   x[tf.newaxis, :, i, tf.newaxis, :], axis=3) + beta0[:, tf.newaxis, :]
            P = tf.sigmoid(logits)
            action_prob = tf.reduce_sum(tf.multiply(P, prob), axis=2)

            ta = ta.write(i, action_prob)

            logits = tf.reduce_sum(a[tf.newaxis, :, i, tf.newaxis, tf.newaxis, :] *
                                   rho[:, tf.newaxis, :, tf.newaxis, :], axis=4) + mu
            Q = tf.stack([
                tf.sigmoid(logits[:, :, :, 0]),
                tf.sigmoid(logits[:, :, :, 1]) -
                tf.sigmoid(logits[:, :, :, 0]),
                1 - tf.sigmoid(logits[:, :, :, 1])
            ], axis=3)
            # Q = tf.nn.softmax(logits, axis=3)
            prob = tf.reduce_sum(
                tf.multiply(Q, prob[:, :, :, tf.newaxis]),
                axis=2
            )
            return i + 1, prob, ta

        i = tf.constant(0)
        i, prob, ta = tf.while_loop(lambda i, *args: tf.less(i, T),
                                    loop_body, [i, prob, ta])

        action_prob = ta.stack()
        action_prob = tf.transpose(action_prob, [1, 2, 0])

        y = zs.Bernoulli('y', tf.log(action_prob) -
                         tf.log(1 - action_prob), group_ndims=2)

    return model


if __name__ == '__main__':
    tf.set_random_seed(42)

    a_it, x_it, y_it, z_it = read_data()
    a_it = a_it[:, :, [0, 1]]
    # a_it[:, :, 2] = (a_it[:, :, 2] - 1987) / 8
    x_it = (x_it - 18)
    T = 23
    n_batch = a_it.shape[0]

    n_chains = 10
    n_burnin = 10000
    n_sample = 1000
    n_leapfrogs = 10

    x = tf.placeholder(tf.float32, shape=[n_batch, T, X_DIM])
    a = tf.placeholder(tf.float32, shape=[n_batch, T, A_DIM])
    z = tf.placeholder(tf.float32, shape=[n_batch, Z_DIM])

    def log_joint(observed, names=['y', 'beta', 'beta0', 'delta', 'rho']):
        model = hmm(observed, x, a, z, n_batch, T, n_chains)
        return sum(map(lambda name: model.local_log_prob(name), names))

    y = tf.placeholder(tf.int32, shape=[n_batch, T])

    std = 0.01
    beta = tf.get_variable('beta', shape=[n_chains, 3], trainable=False,
                           initializer=tf.random_normal_initializer(0, std))
    beta0 = tf.get_variable('beta0', shape=[n_chains, 3], trainable=False,
                            initializer=tf.random_normal_initializer(0, std))
    delta = tf.get_variable('delta', shape=[n_chains, 3, 2, Z_DIM], trainable=False,
                            initializer=tf.random_normal_initializer(0, std))
    rho = tf.get_variable('rho', shape=[n_chains, 3, A_DIM], trainable=False,
                          initializer=tf.random_normal_initializer(0, std))

    adapt_step_size = tf.placeholder(
        tf.bool, shape=[], name='adapt_step_size')
    adapt_mass = tf.placeholder(tf.bool, shape=[], name='adapt_mass')
    hmc = zs.HMC(step_size=1e-3, n_leapfrogs=n_leapfrogs,
                 adapt_step_size=adapt_step_size,
                 adapt_mass=adapt_mass,
                 target_acceptance_rate=0.8)

    sample_op, hmc_info = hmc.sample(
        log_joint, {'y': y}, {'beta': beta, 'beta0': beta0, 'delta': delta, 'rho': rho})

    lr = tf.placeholder(tf.float32, name='lr')
    loss_op = -tf.reduce_mean(log_joint({'y': y, 'beta': beta, 'beta0': beta0,
                                         'delta': delta, 'rho': rho}))
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(
        loss_op, var_list=[beta, beta0, delta, rho])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # lr_value = 1e-2
    # for i in range(1000):
    #     if i > 100:
    #         lr_value = 1e-3
    #     if i > 500:
    #         lr_value = 1e-4
    #     _, loss = sess.run([train_op, loss_op], feed_dict={
    #                        lr: lr_value, y: y_it, x: x_it, a: a_it, z: z_it})
    #     print('Iter {}: Loss = {}'.format(i, loss))

    print('Burning...')
    for i in range(n_burnin):
        _, log_prob, acc, ss = sess.run([sample_op,
                                         hmc_info.log_prob,
                                         hmc_info.acceptance_rate, hmc_info.updated_step_size],
                                        feed_dict={adapt_step_size: i < n_burnin - n_sample,
                                                   adapt_mass: i < n_burnin - n_sample,
                                                   y: y_it, x: x_it, a: a_it, z: z_it})
        if i % 10 == 0:
            print('Sample {}: Log prob = {:.3f} ({:.3f}), Acceptance rate = {:.3f}, updated step size = {:.3e}'
                  .format(i, log_prob.mean(), log_prob.std(), np.mean(acc), ss))

    beta_samples = []
    beta0_samples = []
    delta_samples = []
    rho_samples = []

    print('Sampling....')
    for i in range(n_sample):
        _, log_prob, beta_sample, \
            beta0_sample, delta_sample, rho_sample, \
            acc, ss = sess.run([sample_op,
                                hmc_info.log_prob,
                                hmc_info.samples['beta'], hmc_info.samples['beta0'],
                                hmc_info.samples['delta'], hmc_info.samples['rho'],
                                hmc_info.acceptance_rate, hmc_info.updated_step_size],
                               feed_dict={adapt_step_size: False,
                                          adapt_mass: False,
                                          y: y_it, x: x_it, a: a_it, z: z_it})
        if i % 10 == 0:
            print('Sample {}: Log prob = {:.3f} ({:.3f}), Acceptance rate = {:.3f}, updated step size = {:.3e}'
                  .format(i, log_prob.mean(), log_prob.std(), np.mean(acc), ss))
        beta_samples.append(beta_sample)
        beta0_samples.append(beta0_sample)
        delta_samples.append(delta_sample)
        rho_samples.append(rho_sample)

    print('Finished.')

    print(log_prob)

    log_prob_data = sess.run(log_joint({'y': y, 'beta': beta, 'beta0': beta0,
                                        'delta': delta, 'rho': rho}, ['y']),
                             feed_dict={y: y_it, x: x_it, a: a_it, z: z_it})
    print(log_prob_data)

    sess.close()

    beta = np.stack(beta_samples, axis=1)
    beta0 = np.stack(beta0_samples, axis=1)
    delta = np.stack(delta_samples, axis=1)
    rho = np.stack(rho_samples, axis=1)

    np.savez_compressed('param.npz', beta=beta,
                        beta0=beta0, delta=delta, rho=rho)

    print('beta mean = {}'.format(np.mean(beta, axis=1)))
    print('beta stdev = {}'.format(np.std(beta, axis=1)))

    print('beta0 mean = {}'.format(np.mean(beta0, axis=1)))
    print('beta0 stdev = {}'.format(np.std(beta0, axis=1)))

    # print('delta mean = {}'.format(np.mean(delta, axis=0)))
    # print('delta stdev = {}'.format(np.std(delta, axis=0)))

    print('rho mean = {}'.format(np.mean(rho, axis=1)))
    print('rho stdev = {}'.format(np.std(rho, axis=1)))
