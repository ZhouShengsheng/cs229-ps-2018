import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    import matplotlib.pyplot as plt
    # Search tau_values for the best tau (lowest MSE on the validation set)
    best_tau = None
    best_mse = float('inf')
    for tau in tau_values:
        model = LocallyWeightedLinearRegression(tau)
        model.fit(x_train, y_train)
        x_val, y_val = util.load_dataset(valid_path, add_intercept=True)
        y_pred= model.predict(x_val)
        mse = ((y_val - y_pred) ** 2).mean()
        # plot
        plt.figure()
        plt.title('$tau = {}$'.format(tau))
        plt.plot(x_train, y_train, 'bx')
        plt.plot(x_val, y_pred, 'ro')
        plt.savefig('output/tau_{}.png'.format(tau))
        if mse < best_mse:
            best_mse = mse
            best_tau = tau
    # Fit a LWR model with the best tau value
    print('tau: {}, mse: {}'.format(best_tau, best_mse))
    model = LocallyWeightedLinearRegression(best_tau)
    model.fit(x_train, y_train)
    # Run on the test set to get the MSE value
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    y_pred = model.predict(x_test)
    mse = ((y_test - y_pred) ** 2).mean()
    print('test mse: {}'.format(mse))

    # Save test set predictions to pred_path
    np.savetxt(pred_path, y_pred)
    # P
    plt.figure()
    plt.title('$tau = {}$'.format(tau))
    plt.plot(x_train, y_train, 'bx')
    plt.plot(x_test, y_pred, 'ro')
    plt.savefig('output/final.png')
    # *** END CODE HERE ***
