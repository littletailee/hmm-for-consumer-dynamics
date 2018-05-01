import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import zhusuan as zs


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

    # flatten f X was 1D
    if len(X.shape) == 1:
        p = p.flatten()

    return p


def hmm(observed, x, a, z, n_batch, n_steps, n_particles):
    X_DIM = 1
    A_DIM = 3
    with zs.BayesianNet(observed=observed) as model:
        beta = zs.Normal('beta', mean=[0., 0., 0.],
                         std=10., n_samples=n_particles, group_ndims=1)
        beta0 = zs.Normal('beta0', mean=[0., 0., 0.],
                          std=10., n_samples=n_particles, group_ndims=1)
        delta = zs.Normal('delta', mean=[[[0.] * 3] * 3] * 3, std=10.,
                          n_samples=n_particles, group_ndims=3)
        epsilon = zs.Normal('epsilon', mean=[[[0.] * 3] * 3] * n_batch, std=0.01,
                            n_samples=n_particles, group_ndims=3)
        rho = zs.Normal('rho', mean=[[[0.] * A_DIM] * 3] * 3, std=10.,
                        n_samples=n_particles, group_ndims=3)

        pi = np.array([1., 0., 0.]).astype(np.float32)

        # [np, nb, 3, 3]
        mu = tf.reduce_sum(tf.multiply(
            tf.expand_dims(delta, axis=1),
            z[tf.newaxis, :, tf.newaxis, tf.newaxis, :]
        ), axis=4)

        # Q = np.array([[0.3, 0.3, 0.4], [0.3, 0.3, 0.4],
        #               [0.3, 0.3, 0.4]]).astype(np.float32)

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
                                   rho[:, tf.newaxis, :, :, :], axis=4) + mu
            Q = tf.nn.softmax(logits, axis=3)
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
    T = 23
    n_batch = a_it.shape[0]

    n_chains = 1
    n_iters = 1000
    burnin = n_iters // 2
    n_leapfrogs = 10

    x = tf.placeholder(tf.float32, shape=[n_batch, T, 1])
    a = tf.placeholder(tf.float32, shape=[n_batch, T, 3])
    z = tf.placeholder(tf.float32, shape=[n_batch, 3])

    def log_joint(observed):
        model = hmm(observed, x, a, z, n_batch, T, n_chains)
        # names = ['y', 'beta', 'beta0', 'delta', 'rho']
        names = ['y']
        return sum(map(lambda name: model.local_log_prob(name), names))

    adapt_step_size = tf.placeholder(
        tf.bool, shape=[], name='adapt_step_size')
    adapt_mass = tf.placeholder(tf.bool, shape=[], name='adapt_mass')
    hmc = zs.HMC(step_size=1e-3, n_leapfrogs=n_leapfrogs,
                 adapt_step_size=adapt_step_size,
                 adapt_mass=adapt_mass,
                 target_acceptance_rate=0.8)

    y = tf.placeholder(tf.int32, shape=[n_batch, T])

    beta = tf.Variable(tf.zeros([n_chains, 3]), trainable=False, name='beta')
    beta0 = tf.Variable(tf.zeros([n_chains, 3]), trainable=False, name='beta0')
    delta = tf.Variable(tf.zeros([n_chains, 3, 3, 3]),
                        trainable=False, name='delta')
    rho = tf.Variable(tf.zeros([n_chains, 3, 3, 3]),
                      trainable=False, name='rho')

    sample_op, hmc_info = hmc.sample(
        log_joint, {'y': y}, {'beta': beta, 'beta0': beta0, 'delta': delta, 'rho': rho})

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        beta_samples = []
        beta0_samples = []
        delta_samples = []
        rho_samples = []
        print('Sampling...')
        for i in range(n_iters):
            _, log_prob, beta_sample, beta0_sample, delta_sample, rho_sample, acc, ss = sess.run(
                [sample_op,
                 hmc_info.log_prob,
                 hmc_info.samples['beta'], hmc_info.samples['beta0'],
                 hmc_info.samples['delta'], hmc_info.samples['rho'],
                 hmc_info.acceptance_rate, hmc_info.updated_step_size],
                feed_dict={adapt_step_size: i < burnin // 2,
                           adapt_mass: i < burnin // 2,
                           y: y_it, x: x_it, a: a_it, z: z_it})
            if i % 10 == 0:
                print('Sample {}: Log prob = {}, Acceptance rate = {:.3f}, updated step size = {:.3E}'
                      .format(i, log_prob, np.mean(acc), ss))
            if i >= burnin:
                beta_samples.append(beta_sample)
                beta0_samples.append(beta0_sample)
                delta_samples.append(delta_sample)
                rho_samples.append(rho_sample)
        print('Finished.')
        beta = np.vstack(beta_samples)
        beta0 = np.vstack(beta0_samples)
        delta = np.vstack(delta_samples)
        rho = np.vstack(rho_samples)

    print('beta mean = {}'.format(np.mean(beta, axis=0)))
    print('beta stdev = {}'.format(np.std(beta, axis=0)))

    print('beta0 mean = {}'.format(np.mean(beta0, axis=0)))
    print('beta0 stdev = {}'.format(np.std(beta0, axis=0)))

    print('delta mean = {}'.format(np.mean(delta, axis=0)))
    print('delta stdev = {}'.format(np.std(delta, axis=0)))

    print('rho mean = {}'.format(np.mean(rho, axis=0)))
    print('rho stdev = {}'.format(np.std(rho, axis=0)))
