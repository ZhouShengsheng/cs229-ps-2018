# Important note: you do not have to modify this file for your homework.

import util
import numpy as np


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    m, n = X.shape

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * (X.T.dot(probs * Y))

    return grad


def logistic_regression(X, Y):
    """Train a logistic regression model"""
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 1

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta - learning_rate * grad
        if i % 10000 == 0:
            DEBUG = False
            if DEBUG:
                from util import plot
                from matplotlib import pyplot as plt
                plot(X, (Y == 1), theta, 'output/{}.png'.format(i))
            print('Finished %d iterations' % i)
            # print(np.linalg.norm(prev_theta - theta))
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
    return


def main():
    # plot
    DEBUG = False
    if DEBUG:
        from util import plot_points
        from matplotlib import pyplot as plt
        plt.figure()
        Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=False)
        Ya = (Ya == 1).astype(np.float)
        plot_points(Xa, Ya)

        plt.figure()
        Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=False)
        Yb = (Yb == 1).astype(np.float)
        plot_points(Xb, Yb)
        plt.show()
        import sys
        sys.exit()
    
    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=True)
    logistic_regression(Xa, Ya)

    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=True)
    logistic_regression(Xb, Yb)


if __name__ == '__main__':
    main()
