from __future__ import annotations
from typing import NoReturn
from . import LinearRegression
from ...base import BaseEstimator
import numpy as np
from IMLearn.metrics import loss_functions as ls

class PolynomialFitting(LinearRegression):  # todo what the fuck is the story of this BaseEstimator?????
    """
    Polynomial Fitting using Least Squares estimation
    """

    def __init__(self, k: int) -> PolynomialFitting:
        """
        Instantiate a polynomial fitting estimator

        Parameters
        ----------
        k : int
            Degree of polynomial to fit
        """
        self.k = k
        super().__init__()

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to polynomial transformed samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        mat = self.__transform(X)
        trans_mat = np.transpose(mat)
        self.coefs_ = np.linalg.inv(trans_mat @ mat) @ trans_mat @ y

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.__transform(self.__transform(X)) @ self.coefs_

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """


        return super()._loss(self.__transform(X), y)

    def __transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform given input according to the univariate polynomial transformation

        Parameters
        ----------
        X: ndarray of shape (n_samples,)

        Returns
        -------
        transformed: ndarray of shape (n_samples, k+1)
            Vandermonde matrix of given samples up to degree k
        """
        return np.vander(X, self.k + 1)
