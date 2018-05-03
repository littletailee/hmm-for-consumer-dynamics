import numpy as np

from main import read_data


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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


def main():
    a_it, x_it, y_it, z_it = read_data()
    a_it = a_it[:, :, [0, 1]]
    # a_it[:, :, 2] = (a_it[:, :, 2] - 1987) / 8
    x_it = (x_it - 18)
    T = 23
    n_batch = a_it.shape[0]

    data = np.load('param1-10000-2.npz')
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
        q = softmax(
            np.concatenate([logits, np.zeros_like(
                logits[:, :, 0, np.newaxis])], axis=2),
            axis=2).mean(axis=0)
        # q = np.stack([
        #     sigmoid(logits[:, :, 0]),
        #     sigmoid(logits[:, :, 1]) -
        #     sigmoid(logits[:, :, 0]),
        #     1 - sigmoid(logits[:, :, 1])
        # ], axis=2).mean(axis=0)
        qs.append(q)

    q_mean = np.mean(qs, axis=0)
    print(q_mean)


if __name__ == '__main__':
    main()
