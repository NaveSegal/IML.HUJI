import numpy as np
import pandas as pd


def mean_square_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate MSE loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    MSE of given predictions
    """
    return float(np.mean((y_true - y_pred) ** 2))  # todo check if we need to specify np,mean to return float, not arr


def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> float:
    """
    Calculate misclassification loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    normalize: bool, default = True
        Normalize by number of samples or not

    Returns
    -------
    Misclassification of given predictions
    """
    n = len(y_true)
    df = pd.DataFrame()
    df["y_pred"] = y_pred
    df["y_true"] = y_true
    s = (df.y_pred.ne(df.y_true)
         .rename({True: 1, False: 0}))
    res = sum(s) / n
    return res

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Accuracy of given predictions
    """
    n = len(y_true)
    df = pd.DataFrame()
    df["y_pred"] = y_pred
    df["y_true"] = y_true
    s = (df.y_pred.eq(df.y_true)
         .rename({True: 1, False: 0}))
    res = sum(s) / n
    return res


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the cross entropy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Cross entropy of given predictions
    """
    return 0.314 #todo implement
