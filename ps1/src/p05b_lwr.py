import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    model = LocallyWeightedLinearRegression(0.5)
    model.fit(x_train, y_train)
    # Get MSE value on the validation set
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_val)
    mse = ((y_pred - y_val) ** 2).mean()
    print(mse)
    
    # Plot validation predictions on top of training set
    # No need to save anything
    # Plot data
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(x_train, y_train, 'bx')
    plt.plot(x_val, y_pred, 'ro')
    # plt.show()
    
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        from numpy.linalg import inv, norm
        # gaussian
        m, n = x.shape
        g = lambda x: np.exp(- (x ** 2) / (2 * self.tau ** 2))
        # compute W for each x
        # x: (m, n)
        # self.x: (m0, n)
        # w: (m, m0, m0). compute each theta
        w = g(norm(self.x[None] - x[:, None], axis=2))
        # w = np.diag(w)
        y_pred = np.zeros(m)
        for i, W in enumerate(w):
            W = np.diag(W)
            theta = inv(self.x.T.dot(W).dot(self.x)).dot(self.x.T).dot(W).dot(self.y)
            # make prediction
            y_pred[i] = x[i].dot(theta)
        
        return y_pred
        
        
        # *** END CODE HERE ***
