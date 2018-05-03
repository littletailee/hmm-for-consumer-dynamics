import numpy as np

from main import read_data


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    a_it, x_it, y_it, z_it = read_data()
    a_it = a_it[:, :, [0, 1]]
    # a_it[:, :, 2] = (a_it[:, :, 2] - 1987) / 8
    x_it = (x_it - 18)
    T = 23
    n_batch = a_it.shape[0]

    data = np.load('param-10000-5.npz')
    delta = data['delta']
    rho = data['rho']

    delta = np.vstack(delta).mean(axis=0)
    rho = np.vstack(rho).mean(axis=0)

    mu = np.sum(
        delta[np.newaxis, :, :] *
        z_it[:, np.newaxis, np.newaxis, :], axis=3)

    qs = []

    for i in range(T):
        logits = np.sum(a_it[:, i, np.newaxis, np.newaxis, :] *
                        rho[np.newaxis, :, np.newaxis, :], axis=3) + mu
        q = np.stack([
            sigmoid(logits[:, :, 0]),
            sigmoid(logits[:, :, 1]) -
            sigmoid(logits[:, :, 0]),
            1 - sigmoid(logits[:, :, 1])
        ], axis=2).mean(axis=0)
        qs.append(q)

    q_mean = np.mean(qs, axis=0)
    print(q_mean)


if __name__ == '__main__':
    main()
