from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet
import plotly.graph_objects as go
import plotly.express as px


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters
        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator
        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.
        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples
        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data
        Returns
        -------
        self : returns an instance of self.
        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = X.mean()
        if self.biased_:
            self.var_ = ((X - self.mu_) ** 2).sum() / (len(X))
        else:
            self.var_ = ((X - self.mu_) ** 2).sum() / (len(X) - 1)

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators
        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for
        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)
        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        return (1 / (self.mu_ * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((X - self.mu_) ** 2 / self.var_))

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model
        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with
        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """

        if not sigma:
            return -((X - mu) ** 2).sum()

        return (-1 / (2 * sigma)) * ((X - mu) ** 2).sum()


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """

    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator
        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.
        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples
        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data
        Returns
        -------
        self : returns an instance of self
        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """

        self.mu_ = np.mean(X, axis=0)  # todo check if axis=1 should be axis=0 or opposite
        self.cov_ = np.cov(X, rowvar=False)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators
        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for
        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)
        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        invert = np.linalg.inv(self.cov_)
        deter = np.linalg.det(2 * np.pi * self.cov_) ** (-0.5)

        res = np.array([deter * np.exp(-0.5 * (np.transpose(x - self.mu_) @ invert @ (x - self.mu_))) for x in X])
        return res

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model
        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with
        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """
        inver = np.linalg.inv(cov)
        deter = np.linalg.det(cov)

        def single_loglikelihood(X_minus_mu):
            return -0.5 * (np.log(deter) + X_minus_mu.T.dot(inver).dot(X_minus_mu) + 2 * np.log(2 * np.pi))

        return (np.apply_along_axis(single_loglikelihood, 1, X - mu)).sum()


def univariate_gaussian_sample_generator_and_fitting(mu=10, var=1, size=1000):
    X = np.random.normal(mu, var, size)
    uni_gaus = UnivariateGaussian().fit(X)
    print(f"({uni_gaus.mu_}, {uni_gaus.var_})")

    mu_dist_lst = [abs(UnivariateGaussian().fit(X[:amount]).mu_ - uni_gaus.mu_) for amount in
                   range(10, size, 10)]

    fig = px.scatter(x=range(10, size, 10), y=mu_dist_lst, labels=dict(x="Sample Size", y="Distance from MU"))
    fig.show()

    pdf_dict = {X[indx]: value for indx, value in enumerate(uni_gaus.pdf(X))}

    fig2 = px.scatter(x=pdf_dict.keys(), y=pdf_dict.values(), labels=dict(x="Sample Values", y="PDF(value)"))
    fig2.show()


def connect_mu_to_likelihood_values(amount, log_likelihoods):
    res = {}
    for x_index, x in enumerate(np.linspace(-10, 10, amount)):
        for y_index, y in enumerate(np.linspace(-10, 10, amount)):
            res[(x, y)] = log_likelihoods[x_index][y_index]
    return res


def multivariate_gaussian_sample_generator_and_fitting():
    #   Q 4
    print("Q 4")
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    amount = 1000
    samples = np.random.multivariate_normal(mu, cov, amount)
    multi_gaus = MultivariateGaussian().fit(samples)
    print("multi_gaus.mu_:")
    print(multi_gaus.mu_)
    print("multi_gaus.cov_:")
    print(multi_gaus.cov_)

    #   Q 5
    print("Q 5")
    amount = 200  # todo change to 200
    heatmap = [[multi_gaus.log_likelihood(np.array([y, 0, x, 0]), cov, samples) for x in np.linspace(-10, 10, amount)]
               for y in np.linspace(-10, 10, amount)]

    print("yo yo!")

    fig = go.Figure(data=go.Heatmap(z=heatmap, x=np.linspace(-10, 10, amount), y=np.linspace(-10, 10, amount)))
    fig.show()
    print("Yahooo!")

    heatmap_dict = connect_mu_to_likelihood_values(amount, heatmap)

    max_mu = []
    max_likelihood = -99999999999.0
    for mu, likelihood in heatmap_dict.items():
        if likelihood >= max_likelihood:
            max_mu, max_likelihood = mu, likelihood

    #   Q 6
    print("the max log likelihood is: ", round(max_likelihood, 3))  # todo should we del this,?
    print("the values of mu that coresponse to it are: ", f" {round(max_mu[0], 4)}, {round(max_mu[1], 4)}")


def function_for_the_moodle_ex1():
    print("Yes Please Starts:")
    X = np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4, 1, 2, 1,
                  -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2])
    uni_gaus = UnivariateGaussian().fit(X)
    print(f"({uni_gaus.mu_}, {uni_gaus.var_})")
    print(uni_gaus.log_likelihood(1, 1, X))
    print(uni_gaus.log_likelihood(10, 1, X))


if __name__ == '__main__':
    # np.random.seed(0)
    # print("Here We Start! =)\n\n")
    # univariate_gaussian_sample_generator_and_fitting()
    # multivariate_gaussian_sample_generator_and_fitting()
    # function_for_the_moodle_ex1()

    pass
