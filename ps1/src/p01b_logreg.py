import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    
    # Train a logistic regression classifier
    model = LogisticRegression(eps=1e-5)
    model.fit(x_train, y_train)
    
    # Plot decision boundary on top of validation set set
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_val)
    util.plot(x_val, y_val, model.theta, '{}.png'.format(pred_path))
    
    # Use np.savetxt to save predictions on eval set to pred_path
    np.savetxt(pred_path, y_pred)
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver."""

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        
        g = lambda x: 1 / (1 + np.exp(-x))
        m, n = x.shape
        
        # initialize theta
        if self.theta is None:
            self.theta = np.zeros(n)
        
        # optimize theta
        # for i in range(self.max_iter):
        while True:
            theta = self.theta
            # compute J
            J = - (1 / m) * (y - g(x.dot(theta))).dot(x)
            
            # compute H
            x_theta = x.dot(theta)
            H = (1 / m) * g(x_theta).dot(g(1 - x_theta)) * (x.T).dot(x)
            H_inv = np.linalg.inv(H)
            
            # update
            self.theta = theta - H_inv.dot(J)
            
            # if norm is small, terminate
            if np.linalg.norm(self.theta - theta, ord=1) < self.eps:
                break
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        # compute probability
        g = lambda x: 1 / (1 + np.exp(-x))
        preds = g(x.dot(self.theta))
        
        return preds
        # *** END CODE HERE ***
