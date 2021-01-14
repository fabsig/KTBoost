""" Boosting with Trees and Kernel Regressors as base learners

This module contains methods for boosting for both classification and 
regression. Newton boosting, gradient boosting, as well as a hybrid variant of
the two is supported (see parameter 'update_step'). Regression trees,
kernel Ridge regressors, and a combination of the two are supported as base
learners (see parameter 'base_learner').

The module structure is the following:

- The ``BaseBoosting`` base class implements a common ``fit`` method
  for all the estimators in the module. Regression and classification
  only differ in the concrete ``LossFunction`` used.

- ``BoostingClassifier`` implements boosting for classification problems.

- ``BoostingRegressor`` implements boosting for regression problems.
"""

# Authors: Fabio Sigrist (fabiosigrist@gmail.com)
# (and authors from the scikit-learn implementation of boosting
#  as this module extends the scikit-learn functionality on boosting)
# License: BSD 3 clause

from __future__ import print_function
from __future__ import division

from abc import ABCMeta
from abc import abstractmethod

from sklearn.ensemble import BaseEnsemble
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
import six as six

from sklearn.ensemble._gradient_boosting import predict_stages
from sklearn.ensemble._gradient_boosting import predict_stage
from sklearn.ensemble._gradient_boosting import _random_sample_mask

import numbers
import numpy as np

import math as math
from scipy.stats import norm

from scipy import stats
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from scipy.special import expit

from time import time
from sklearn.model_selection import train_test_split
from .kernel_ridge import KernelRidge
from .tree import DecisionTreeRegressor
from scipy import sparse as sparse
from scipy.sparse import linalg as sp_linalg
from scipy import linalg

from sklearn.tree._tree import DTYPE
from sklearn.tree._tree import TREE_LEAF

from sklearn import preprocessing

from sklearn.utils import check_random_state
from sklearn.utils import check_array
from sklearn.utils import check_X_y
from sklearn.utils import column_or_1d
from sklearn.utils import check_consistent_length
from sklearn.utils import deprecated
from sklearn.utils.stats import _weighted_percentile
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.exceptions import NotFittedError

from scipy.special import logsumexp

import warnings

import matplotlib.pyplot as plt
    
MAX_VAL_PRED=1e30
MAX_VAL_LOGPRED=19*np.log(10)
MIN_VAL_HESSIAN=1e-20

class QuantileEstimator(object):
    """An estimator predicting the alpha-quantile of the training targets."""
    def __init__(self, alpha=0.9):
        if not 0 < alpha < 1.0:
            raise ValueError("`alpha` must be in (0, 1.0) but was %r" % alpha)
        self.alpha = alpha

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            self.quantile = stats.scoreatpercentile(y, self.alpha * 100.0)
        else:
            self.quantile = _weighted_percentile(y, sample_weight,
                                                 self.alpha * 100.0)

    def predict(self, X):
        check_is_fitted(self, 'quantile')

        y = np.empty((X.shape[0], 1), dtype=np.float64)
        y.fill(self.quantile)
        return y


class MeanEstimator(object):
    """An estimator predicting the mean of the training targets."""
    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            self.mean = np.mean(y)
        else:
            self.mean = np.average(y, weights=sample_weight)

    def predict(self, X):
        check_is_fitted(self, 'mean')

        y = np.empty((X.shape[0], 1), dtype=np.float64)
        y.fill(self.mean)
        return y


class MeanLogScaleEstimator(object):
    """An estimator predicting the mean and the logarithm of the standard 
    deviation (=scale) of the training targets
    """
    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            self.mean = np.mean(y)
            self.logstd = 0.5*np.log(np.average((y-self.mean)**2))
        else:
            self.mean = np.average(y, weights=sample_weight)
            self.logstd = 0.5*np.log(np.average((y-self.mean)**2,
                                                weights=sample_weight))

    def predict(self, X):
        check_is_fitted(self, 'mean')
        check_is_fitted(self, 'logstd')

        y = np.empty((X.shape[0], 2), dtype=np.float64)
        y[:,0] = self.mean
        y[:,1] = self.logstd
        return y


class LogMeanEstimator(object):
    """An estimator for the logarithm of the mean of the training targets."""
    def fit(self, X, y, sample_weight=None):
        if (y < 0).any():
            raise ValueError('y contains negative numbers.')
        if sample_weight is None:
            self.logmean = np.log(np.max([np.mean(y),1e-20]))
        else:
            self.logmean = np.log(np.max([np.average(y, weights=sample_weight),
                                          1e-20]))

    def predict(self, X):
        check_is_fitted(self, 'logmean')

        y = np.empty((X.shape[0], 1), dtype=np.float64)
        y.fill(self.logmean)
        return y


class LogOddsEstimator(object):
    """An estimator predicting the log odds ratio."""
    scale = 1.0

    def fit(self, X, y, sample_weight=None):
        # pre-cond: pos, neg are encoded as 1, 0
        if sample_weight is None:
            pos = np.sum(y)
            neg = y.shape[0] - pos
        else:
            pos = np.sum(sample_weight * y)
            neg = np.sum(sample_weight * (1 - y))

        if neg == 0 or pos == 0:
            raise ValueError('y contains non binary labels.')
        self.prior = self.scale * np.log(pos / neg)

    def predict(self, X):
        check_is_fitted(self, 'prior')

        y = np.empty((X.shape[0], 1), dtype=np.float64)
        y.fill(self.prior)
        return y


class ScaledLogOddsEstimator(LogOddsEstimator):
    """Log odds ratio scaled by 0.5 -- for exponential loss. """
    scale = 0.5


class PriorProbabilityEstimator(object):
    """An estimator predicting the probability of each
    class in the training data.
    """
    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones_like(y, dtype=np.float64)
        class_counts = np.bincount(y, weights=sample_weight)
        self.priors = class_counts / class_counts.sum()

    def predict(self, X):
        check_is_fitted(self, 'priors')

        y = np.empty((X.shape[0], self.priors.shape[0]), dtype=np.float64)
        y[:] = self.priors
        return y


class ZeroEstimator(object):
    """An estimator that simply predicts zero. """

    def fit(self, X, y, sample_weight=None):
        if np.issubdtype(y.dtype, np.signedinteger):
            # classification
            self.n_classes = np.unique(y).shape[0]
            if self.n_classes == 2:
                self.n_classes = 1
        else:
            # regression
            self.n_classes = 1

    def predict(self, X):
        check_is_fitted(self, 'n_classes')

        y = np.empty((X.shape[0], self.n_classes), dtype=np.float64)
        y.fill(0.0)
        return y


class TobitEstimator(object):
    """An estimator for the mean of the latent variable
    of the Tobit model."""
    def __init__(self, sigma=1, yl=0, yu=1):
        if not 0 < sigma:
            raise ValueError("`sigma` must be larger than 0 but was %r"
                             % sigma)
        self.sigma = sigma
        if not yl < yu:
            raise ValueError("`yl` must be smaller than `yu`")
        self.yl = yl
        self.yu = yu

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            self.mean = np.mean(y)
        else:
            self.mean = np.average(y, weights=sample_weight)

    def predict(self, X):
        check_is_fitted(self, 'mean')

        y = np.empty((X.shape[0], 1), dtype=np.float64)
        y.fill(self.mean)
        return y


class GammaEstimator(object):
    """An estimator for the logarithm of lambda of the Gamma model."""
    def __init__(self, gamma=1):
        if not 0 < gamma:
            raise ValueError("`sigma` must be larger than 0 but was %r"
                             % gamma)
        self.gamma = gamma

    def fit(self, X, y, sample_weight=None):
        if (y < 0).any():
            raise ValueError('y contains negative numbers.')
        if sample_weight is None:
            self.loglambda = (np.log(self.gamma)
                            - np.log(np.max([np.mean(y),1e-20])))
        else:
            self.loglambda = (np.log(self.gamma)
                            - np.log(np.max([np.average(y,
                                                        weights=sample_weight),
                                              1e-20])))

    def predict(self, X):
        check_is_fitted(self, 'loglambda')

        y = np.empty((X.shape[0], 1), dtype=np.float64)
        y.fill(self.loglambda)
        return y


class LossFunction(six.with_metaclass(ABCMeta, object)):
    """Abstract base class for various loss functions.

    Attributes
    ----------
    K : int
        The number of regression trees to be induced;
        1 for regression and binary classification;
        ``n_classes`` for multi-class classification.
    """

    is_multi_class = False

    def __init__(self, n_classes):
        self.K = n_classes

    def init_estimator(self):
        """Default ``init`` estimator for loss function. """
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, y, pred, sample_weight=None):
        """Compute the loss of prediction ``pred`` and ``y``. """

    @abstractmethod
    def negative_gradient(self, y, y_pred, **kargs):
        """Compute the negative gradient.

        Parameters
        ---------
        y : np.ndarray, shape=(n,)
            The target labels.
        y_pred : np.ndarray, shape=(n,):
            The predictions.
        """

    @abstractmethod
    def hessian(self, y, pred, residual, **kargs):
        """Compute the second derivative.

        Parameters
        ---------
        y : np.ndarray, shape=(n,)
            The target labels.
        y_pred : np.ndarray, shape=(n,):
            The predictions.
        residual : np.ndarray, shape=(n,):
            negative gradient.
        """

    def update_terminal_regions(self, tree, X, y, residual, y_pred,
                                sample_weight, sample_mask,
                                learning_rate=1.0, k=0, update_step="hybrid"):
        """Update the terminal regions (=leaves) of the given tree and
        updates the current predictions of the model. Traverses tree
        and invokes template method `_update_terminal_region`.

        Parameters
        ----------
        tree : tree.Tree
            The tree object.
        X : ndarray, shape=(n, m)
            The data array.
        y : ndarray, shape=(n,)
            The target labels.
        residual : ndarray, shape=(n,)
            The residuals (usually the negative gradient).
        y_pred : ndarray, shape=(n,)
            The predictions.
        sample_weight : ndarray, shape=(n,)
            The weight of each sample.
        sample_mask : ndarray, shape=(n,)
            The sample mask to be used.
        learning_rate : float, default=0.1
            learning rate shrinks the contribution of each base learner by
             ``learning_rate``.
        k : int, default 0
            The index of the estimator being updated.
        """
        # compute leaf for each sample in ``X``.
        terminal_regions = tree.apply(X)

        if update_step=="hybrid":
            # mask all which are not in sample mask.
            masked_terminal_regions = terminal_regions.copy()
            masked_terminal_regions[~sample_mask] = -1

            # update each leaf (= perform line search)
            for leaf in np.where(tree.children_left == TREE_LEAF)[0]:
                self._update_terminal_region(tree, masked_terminal_regions,
                                             leaf, X, y, residual,
                                             y_pred[:, k], sample_weight)

        # update predictions (both in-bag and out-of-bag)
        y_pred[:, k] += (learning_rate
                         * tree.value[:, 0, 0].take(terminal_regions, axis=0))

        self.avoid_overflow(y_pred,k)

    @abstractmethod
    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred, sample_weight):
        """Template method for updating terminal regions (=leaves). """

    def avoid_overflow(self, y_pred, k):
        """Constraining predictions to a certain range in order to avoid 
        numerical overflows. 

        Parameters
        ----------
        y_pred : ndarray, shape=(n,)
            The predictions.
        k : int, default 0
            The index of the estimator being updated.    
        """
        
        y_pred[:, k][y_pred[:, k]>MAX_VAL_PRED]=MAX_VAL_PRED
        y_pred[:, k][y_pred[:, k]<-MAX_VAL_PRED]=-MAX_VAL_PRED


class RegressionLossFunction(six.with_metaclass(ABCMeta, LossFunction)):
    """Base class for regression loss functions. """

    def __init__(self, n_classes):
        if n_classes != 1:
            raise ValueError("``n_classes`` must be 1 for regression but "
                             "was %r" % n_classes)
        super(RegressionLossFunction, self).__init__(n_classes)


class LeastSquaresError(RegressionLossFunction):
    """Loss function for least squares (LS) estimation.
    Terminal regions need not to be updated for least squares. """
    def init_estimator(self):
        return MeanEstimator()

    def __call__(self, y, pred, sample_weight=None):
        if sample_weight is None:
            return np.mean((y - pred.ravel()) ** 2.0)
        else:
            return (1.0 / sample_weight.sum() *
                    np.sum(sample_weight * ((y - pred.ravel()) ** 2.0)))

    def negative_gradient(self, y, pred, **kargs):
        return y - pred.ravel()

    def update_terminal_regions(self, tree, X, y, residual, y_pred,
                                sample_weight, sample_mask,
                                learning_rate=1.0, k=0, update_step="hybrid"):
        """Least squares does not need to update terminal regions.

        But it has to update the predictions.
        """
        # update predictions
        y_pred[:, k] += learning_rate * tree.predict(X).ravel()
        self.avoid_overflow(y_pred,k)

    def hessian(self, y, pred, residual, **kargs):
        hessian = np.ones((y.shape[0],), dtype=np.float64)
        return hessian

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred, sample_weight):
        pass


class MeanScaleRegressionLoss(RegressionLossFunction):
    """Loss function for the case where both the mean and the standard 
    deviation (=scale) consist of ensembles of base learners, which are learned 
    using boosting."""
    def __init__(self, n_classes):
        super(MeanScaleRegressionLoss, self).__init__(n_classes)
        self.const = 0.5 * np.log(2 * math.pi)
        self.K = 2
        
    def init_estimator(self):
        return MeanLogScaleEstimator()

    def __call__(self, y, pred, sample_weight=None):
        if sample_weight is None:
            return np.mean(((y - pred[:,0]) ** 2.0)/(2.*np.exp(2.*pred[:,1]))+
                           pred[:,1]+self.const)
        else:
            return (1.0 / sample_weight.sum() *
                    np.sum(sample_weight * (((y - pred[:,0]) ** 2.0)/(2.*np.exp(2.*pred[:,1]))+
                           pred[:,1]+self.const)))

    def negative_gradient(self, y, pred, k=0, **kargs):
        if k==0: return (y - pred[:,0])/np.exp(2.*pred[:,1])
        if k==1: return (((y - pred[:,0]) ** 2.0)/np.exp(2.*pred[:,1])-1.)

    def hessian(self, y, pred, k=0, **kargs):
        """Compute the second derivative """
        if k==0: return 1./np.exp(2.*pred[:,1])
        if k==1: return 2.*((y - pred[:,0]) ** 2.0)/np.exp(2.*pred[:,1])

    def update_terminal_regions(self, tree, X, y, residual, y_pred,
                                sample_weight, sample_mask,
                                learning_rate=1.0, k=0, update_step="hybrid"):
        """Same function as in the base class, but the index k needs also to be
        passed to the function '_update_terminal_region' since the Hessian is
        not equal for the mean and the log standard deviation
        """
        # compute leaf for each sample in ``X``.
        terminal_regions = tree.apply(X)

        if update_step=="hybrid":
            # mask all which are not in sample mask.
            masked_terminal_regions = terminal_regions.copy()
            masked_terminal_regions[~sample_mask] = -1

            # update each leaf (= perform line search)
            for leaf in np.where(tree.children_left == TREE_LEAF)[0]:
                self._update_terminal_region(tree, masked_terminal_regions,
                                             leaf, X, y, residual,
                                             y_pred, sample_weight, k)

        # update predictions (both in-bag and out-of-bag)
        y_pred[:, k] += (learning_rate
                         * tree.value[:, 0, 0].take(terminal_regions, axis=0))
        self.avoid_overflow(y_pred,k)

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred, sample_weight, k):
        """Make a single Newton-Raphson step. """
        terminal_region = np.where(terminal_regions == leaf)[0]
        residual = residual.take(terminal_region, axis=0)
        y = y.take(terminal_region, axis=0)
        predLeaf = pred.take(terminal_region, axis=0)
        sample_weight = sample_weight.take(terminal_region, axis=0)
        
        numerator = np.sum(sample_weight * residual)
        denominator = np.sum(self.hessian(y=y, pred=predLeaf, k=k))

        # prevents overflow and division by zero
        if abs(denominator) < 1e-150:
            tree.value[leaf, 0, 0] = 0.0
        else:
            tree.value[leaf, 0, 0] = numerator / denominator

    def avoid_overflow(self, y_pred, k):
        if k==0:
            y_pred[:, k][y_pred[:, k]>MAX_VAL_PRED]=MAX_VAL_PRED
            y_pred[:, k][y_pred[:, k]<-MAX_VAL_PRED]=-MAX_VAL_PRED
        if k==1:
            y_pred[:, k][y_pred[:, k]>MAX_VAL_LOGPRED]=MAX_VAL_LOGPRED
            y_pred[:, k][y_pred[:, k]<-MAX_VAL_LOGPRED]=-MAX_VAL_LOGPRED

class LeastAbsoluteError(RegressionLossFunction):
    """Loss function for least absolute deviation (LAD) regression. """
    def init_estimator(self):
        return QuantileEstimator(alpha=0.5)

    def __call__(self, y, pred, sample_weight=None):
        if sample_weight is None:
            return np.abs(y - pred.ravel()).mean()
        else:
            return (1.0 / sample_weight.sum() *
                    np.sum(sample_weight * np.abs(y - pred.ravel())))

    def negative_gradient(self, y, pred, **kargs):
        """1.0 if y - pred > 0.0 else -1.0"""
        pred = pred.ravel()
        return 2.0 * (y - pred > 0.0) - 1.0

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred, sample_weight):
        """LAD updates terminal regions to median estimates. """
        terminal_region = np.where(terminal_regions == leaf)[0]
        sample_weight = sample_weight.take(terminal_region, axis=0)
        diff = y.take(terminal_region, axis=0) - pred.take(terminal_region, axis=0)
        tree.value[leaf, 0, 0] = _weighted_percentile(diff, sample_weight, percentile=50)

    def hessian(self, y, pred, residual, **kargs):
        """LeastAbsoluteError does not need to calculate the Hessian.
        """


class HuberLossFunction(RegressionLossFunction):
    """Huber loss function for robust regression.

    M-Regression proposed in Friedman 2001.

    References
    ----------
    J. Friedman, Greedy Function Approximation: A Gradient Boosting
    Machine, The Annals of Statistics, Vol. 29, No. 5, 2001.
    """

    def __init__(self, n_classes, alpha=0.9):
        super(HuberLossFunction, self).__init__(n_classes)
        self.alpha = alpha
        self.gamma = None

    def init_estimator(self):
        return QuantileEstimator(alpha=0.5)

    def __call__(self, y, pred, sample_weight=None):
        pred = pred.ravel()
        diff = y - pred
        gamma = self.gamma
        if gamma is None:
            if sample_weight is None:
                gamma = stats.scoreatpercentile(np.abs(diff), self.alpha * 100)
            else:
                gamma = _weighted_percentile(np.abs(diff), sample_weight, self.alpha * 100)

        gamma_mask = np.abs(diff) <= gamma
        if sample_weight is None:
            sq_loss = np.sum(0.5 * diff[gamma_mask] ** 2.0)
            lin_loss = np.sum(gamma * (np.abs(diff[~gamma_mask]) - gamma / 2.0))
            loss = (sq_loss + lin_loss) / y.shape[0]
        else:
            sq_loss = np.sum(0.5 * sample_weight[gamma_mask] * diff[gamma_mask] ** 2.0)
            lin_loss = np.sum(gamma * sample_weight[~gamma_mask] *
                              (np.abs(diff[~gamma_mask]) - gamma / 2.0))
            loss = (sq_loss + lin_loss) / sample_weight.sum()
        return loss

    def negative_gradient(self, y, pred, sample_weight=None, **kargs):
        pred = pred.ravel()
        diff = y - pred
        if sample_weight is None:
            gamma = stats.scoreatpercentile(np.abs(diff), self.alpha * 100)
        else:
            gamma = _weighted_percentile(np.abs(diff), sample_weight, self.alpha * 100)
        gamma_mask = np.abs(diff) <= gamma
        residual = np.zeros((y.shape[0],), dtype=np.float64)
        residual[gamma_mask] = diff[gamma_mask]
        residual[~gamma_mask] = gamma * np.sign(diff[~gamma_mask])
        self.gamma = gamma
        return residual

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred, sample_weight):
        terminal_region = np.where(terminal_regions == leaf)[0]
        sample_weight = sample_weight.take(terminal_region, axis=0)
        gamma = self.gamma
        diff = (y.take(terminal_region, axis=0)
                - pred.take(terminal_region, axis=0))
        median = _weighted_percentile(diff, sample_weight, percentile=50)
        diff_minus_median = diff - median
        tree.value[leaf, 0] = median + np.mean(
            np.sign(diff_minus_median) *
            np.minimum(np.abs(diff_minus_median), gamma))

    def hessian(self, y, pred, residual, **kargs):
        """HuberLossFunction does not need to calculate the Hessian.
        """


class QuantileLossFunction(RegressionLossFunction):
    """Loss function for quantile regression.

    Quantile regression allows to estimate the percentiles
    of the conditional distribution of the target.
    """

    def __init__(self, n_classes, alpha=0.9):
        super(QuantileLossFunction, self).__init__(n_classes)
        self.alpha = alpha
        self.percentile = alpha * 100.0

    def init_estimator(self):
        return QuantileEstimator(self.alpha)

    def __call__(self, y, pred, sample_weight=None):
        pred = pred.ravel()
        diff = y - pred
        alpha = self.alpha

        mask = y > pred
        if sample_weight is None:
            loss = (alpha * diff[mask].sum() -
                    (1.0 - alpha) * diff[~mask].sum()) / y.shape[0]
        else:
            loss = (((alpha * np.sum(sample_weight[mask] * diff[mask]) -
                    (1.0 - alpha) * np.sum(sample_weight[~mask] *
                    diff[~mask])) / sample_weight.sum()))
        return loss

    def negative_gradient(self, y, pred, **kargs):
        alpha = self.alpha
        pred = pred.ravel()
        mask = y > pred
        return (alpha * mask) - ((1.0 - alpha) * ~mask)

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred, sample_weight):
        terminal_region = np.where(terminal_regions == leaf)[0]
        diff = (y.take(terminal_region, axis=0)
                - pred.take(terminal_region, axis=0))
        sample_weight = sample_weight.take(terminal_region, axis=0)

        val = _weighted_percentile(diff, sample_weight, self.percentile)
        tree.value[leaf, 0] = val

    def hessian(self, y, pred, residual, **kargs):
        """QuantileLossFunction does not need to calculate the Hessian.
        """


class TobitLossFunction(RegressionLossFunction):
    """Loss function for the Tobit model.

    The Tobit model is used, for instance, for modeling censored data.

    References
    ----------
    Sigrist, F., & Hirnschall, C. (2019). Grabit: Gradient Tree Boosted Tobit
    Models for Default Prediction. Journal of Banking and Finance
    """

    def __init__(self, n_classes, sigma=1, yl=0, yu=1):
        super(TobitLossFunction, self).__init__(n_classes)
        self.sigma = sigma
        self.yl = yl
        self.yu = yu
        self.const = 0.5 * np.log(2 * math.pi) + np.log(sigma)

    def init_estimator(self):
        return TobitEstimator(self.sigma, self.yl, self.yu)

    def __call__(self, y, pred, sample_weight=None):
        pred = pred.ravel()
        sigma = self.sigma
        yl = self.yl
        yu = self.yu
        const = self.const
        diff = (y - pred) / sigma
        indl = (y == yl)
        indu = (y == yu)
        indmid = (y > yl) & (y < yu)
        if sample_weight is None:
            loss = (np.sum((diff[indmid] ** 2.0)/2 + const)
                    - np.sum(norm.logcdf(diff[indl]))
                    - np.sum(norm.logcdf(-diff[indu])))
        else:
            loss = (((np.sum(sample_weight[indmid]
                     * ((diff[indmid] ** 2.0) / 2 + const))
                     - np.sum(sample_weight[indl] * norm.logcdf(diff[indl]))
                     - np.sum(sample_weight[indu] * norm.logcdf(-diff[indu])))) 
                    / sample_weight.sum())
        return loss

    def negative_gradient(self, y, pred, **kargs):
        pred = pred.ravel()
        sigma = self.sigma
        yl = self.yl
        yu = self.yu
        diff = (y - pred)/sigma
        indl = (y == yl)
        indu = (y == yu)
        indmid = (y > yl) & (y < yu)
        residual = np.zeros((y.shape[0],), dtype=np.float64)
        residual[indl] = (- np.exp(norm.logpdf(diff[indl])
                          - norm.logcdf(diff[indl])) / sigma)
        residual[indmid] = diff[indmid] / sigma
        residual[indu] = (np.exp(norm.logpdf(diff[indu])
                          - norm.logcdf(-diff[indu])) / sigma)
        return (residual)

    def hessian(self, y, pred, residual, **kargs):
        """Compute the second derivative """
        sigma = self.sigma
        sigma2 = self.sigma ** 2
        yl = self.yl
        yu = self.yu
        diff = (y - pred.ravel())/sigma
        indl = (y == yl)
        indu = (y == yu)
        indmid = (y > yl) & (y < yu)
        hessian = np.zeros((y.shape[0],), dtype=np.float64)
        lognpdfl = norm.logpdf(diff[indl])
        logncdfl = norm.logcdf(diff[indl])
        lognpdfu = norm.logpdf(diff[indu])
        logncdfu = norm.logcdf(-diff[indu])
        hessian[indmid] = 1/sigma2
        hessian[indl] = (np.exp(lognpdfl - logncdfl) / sigma2 * diff[indl]
                         + np.exp(2*lognpdfl-2 * logncdfl) / sigma2)
        hessian[indu] = (- np.exp(lognpdfu-logncdfu)/sigma2 * diff[indu]
                         + np.exp(2*lognpdfu-2 * logncdfu) / sigma2)
        return hessian

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred, sample_weight):
        """Make a single Newton-Raphson step.
        """
        terminal_region = np.where(terminal_regions == leaf)[0]
        residual = residual.take(terminal_region, axis=0)
        y_tr = y.take(terminal_region, axis=0)
        sample_weight = sample_weight.take(terminal_region, axis=0)

        pred = pred.ravel()
        sigma = self.sigma
        sigma2 = self.sigma ** 2
        yl = self.yl
        yu = self.yu

        diff = (y_tr - pred.take(terminal_region, axis=0)) / sigma
        indl = (y_tr == yl)
        indu = (y_tr == yu)
        indmid = (y_tr > yl) & (y_tr < yu)
        hessian = np.zeros((y_tr.shape[0],), dtype=np.float64)

        lognpdfl = norm.logpdf(diff[indl])
        logncdfl = norm.logcdf(diff[indl])
        lognpdfu = norm.logpdf(diff[indu])
        logncdfu = norm.logcdf(-diff[indu])

        hessian[indmid] = 1/sigma2
        hessian[indl] = (np.exp(lognpdfl - logncdfl) / sigma2 * diff[indl]
                         + np.exp(2*lognpdfl-2 * logncdfl) / sigma2)
        hessian[indu] = (- np.exp(lognpdfu-logncdfu)/sigma2 * diff[indu]
                         + np.exp(2*lognpdfu-2 * logncdfu) / sigma2)

        numerator = np.sum(sample_weight * residual)
        denominator = np.sum(sample_weight * hessian)

        if denominator < 1e-150:
            tree.value[leaf, 0] = 0.0
        else:
            tree.value[leaf, 0] = numerator / denominator


class PoissonLossFunction(RegressionLossFunction):
    """Loss function for the Poisson model.

    """

    def __init__(self, n_classes):
        super(PoissonLossFunction, self).__init__(n_classes)

    def init_estimator(self):
        return LogMeanEstimator()

    def __call__(self, y, pred, sample_weight=None):
        constants = [np.log(float(math.factorial(yi))) for yi in y]
        pred = pred.ravel()
        if sample_weight is None:
            loss = np.sum(-y * pred + np.exp(pred) + constants)
        else:
            loss = np.sum((-y * pred + np.exp(pred) + constants)
                            * sample_weight)
        return loss

    def negative_gradient(self, y, pred, **kargs):
        return y - np.exp(pred.ravel())

    def hessian(self, y, pred, residual, **kargs):
        """Compute the second derivative """
        return np.exp(pred.ravel())

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred, sample_weight):
        """Make a single Newton-Raphson step.
        """
        terminal_region = np.where(terminal_regions == leaf)[0]
        residual = residual.take(terminal_region, axis=0)
        sample_weight = sample_weight.take(terminal_region, axis=0)
        pred = pred.take(terminal_region, axis=0)

        numerator = np.sum(sample_weight * residual)
        denominator = np.sum(sample_weight * np.exp(pred))

        if abs(denominator) < 1e-150:
            tree.value[leaf, 0] = 0.0
        else:
            tree.value[leaf, 0] = numerator / denominator

    def avoid_overflow(self, y_pred, k):
        y_pred[:, k][y_pred[:, k]>MAX_VAL_LOGPRED]=MAX_VAL_LOGPRED
        y_pred[:, k][y_pred[:, k]<-MAX_VAL_LOGPRED]=-MAX_VAL_LOGPRED


class TweedieLossFunction(RegressionLossFunction):
    """Loss function for the Tweedie model.

    """

    def __init__(self, n_classes, p=1.5):
        super(TweedieLossFunction, self).__init__(n_classes)
        self.p = p
        
    def init_estimator(self):
        return LogMeanEstimator()

    def __call__(self, y, pred, sample_weight=None):
        
        pred = pred.ravel()
        
        if sample_weight is None:
            loss = 2 * np.mean(
                np.power(y, 2-self.p) / ((1-self.p) * (2-self.p)) -
                y * np.exp(pred * (1-self.p)) / (1-self.p) +
                np.exp(pred* (2-self.p)) / (2-self.p)
            )
        else:
            loss = 2 * np.mean((
                    np.power(y, 2-self.p) / ((1-self.p) * (2-self.p)) -
                    y * np.exp(pred * (1-self.p)) / (1-self.p) +
                    np.exp(pred * (2-self.p)) / (2-self.p)
                ) * sample_weight
            )
        return loss

    def negative_gradient(self, y, pred, **kargs):
        pred = pred.ravel()
        return y * np.exp(pred * (1-self.p)) - np.exp(pred * (2-self.p))

    def hessian(self, y, pred, residual, **kargs):
        """Compute the second derivative """
        pred = pred.ravel()
        return (
            - y * (1-self.p) * np.exp(pred*(1-self.p)) + 
            (2-self.p) * np.exp(pred*(2-self.p))
        )

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred, sample_weight):
        """Make a single Newton-Raphson step.
        """
        terminal_region = np.where(terminal_regions == leaf)[0]
        residual = residual.take(terminal_region, axis=0)
        sample_weight = sample_weight.take(terminal_region, axis=0)
        pred = pred.take(terminal_region, axis=0)
        y_tr = y.take(terminal_region, axis=0)
        
        numerator = np.sum(sample_weight * residual)
        denominator = np.sum(sample_weight * (
            - y_tr * (1-self.p) * np.exp(pred*(1-self.p)) + 
            (2-self.p) * np.exp(pred*(2-self.p))
        ))

        if abs(denominator) < 1e-150:
            tree.value[leaf, 0] = 0.0
        else:
            tree.value[leaf, 0] = numerator / denominator

    def avoid_overflow(self, y_pred, k):
        y_pred[:, k][y_pred[:, k]>MAX_VAL_LOGPRED]=MAX_VAL_LOGPRED
        y_pred[:, k][y_pred[:, k]<-MAX_VAL_LOGPRED]=-MAX_VAL_LOGPRED




class GammaLossFunction(RegressionLossFunction):
    """Loss function for the Gamma model.

    """

    def __init__(self, n_classes, gamma=1):
        super(GammaLossFunction, self).__init__(n_classes)
        self.gamma = gamma

    def init_estimator(self):
        return LogMeanEstimator()

    def __call__(self, y, pred, sample_weight=None):
        
        if self.gamma == 1:
            constants = 0
        else:
            constants = (-(self.gamma-1) * np.log(y) + math.lgamma(self.gamma)
                        - self.gamma * np.log(self.gamma))
        pred = pred.ravel()
        if sample_weight is None:
            loss = np.sum(self.gamma * (pred + np.exp(-pred) * y) + constants)
        else:
            loss = np.sum((self.gamma * (pred + np.exp(-pred) * y) + constants)
                            * sample_weight)
        return loss

    def negative_gradient(self, y, pred, **kargs):
        return -self.gamma * (1 - np.exp(-pred.ravel()) * y)

    def hessian(self, y, pred, residual, **kargs):
        """Compute the second derivative """
        return self.gamma * np.exp(-pred.ravel()) * y

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred, sample_weight):
        """Make a single Newton-Raphson step.
        """
        terminal_region = np.where(terminal_regions == leaf)[0]
        residual = residual.take(terminal_region, axis=0)
        sample_weight = sample_weight.take(terminal_region, axis=0)
        pred = pred.take(terminal_region, axis=0)
        y_tr = y.take(terminal_region, axis=0)

        numerator = np.sum(sample_weight * residual)
        denominator = np.sum(sample_weight * self.gamma * np.exp(-pred) * y_tr)
        
        if abs(denominator) < 1e-150:
            tree.value[leaf, 0] = 0.0
        else:
            tree.value[leaf, 0] = numerator / denominator

    def avoid_overflow(self, y_pred, k):
        y_pred[:, k][y_pred[:, k]>MAX_VAL_LOGPRED]=MAX_VAL_LOGPRED
        y_pred[:, k][y_pred[:, k]<-MAX_VAL_LOGPRED]=-MAX_VAL_LOGPRED


class ClassificationLossFunction(six.with_metaclass(ABCMeta, LossFunction)):
    """Base class for classification loss functions. """

    def _score_to_proba(self, score):
        """Template method to convert scores to probabilities.

         the does not support probabilities raises AttributeError.
        """
        raise TypeError('%s does not support predict_proba' % type(self).__name__)

    @abstractmethod
    def _score_to_decision(self, score):
        """Template method to convert scores to decisions.

        Returns int arrays.
        """


class BinomialDeviance(ClassificationLossFunction):
    """Binomial deviance loss function for binary classification.

    Binary classification is a special case; here, we only need to
    fit one tree instead of ``n_classes`` trees.
    """
    def __init__(self, n_classes):
        if n_classes != 2:
            raise ValueError("{0:s} requires 2 classes; got {1:d} class(es)"
                             .format(self.__class__.__name__, n_classes))
        # we only need to fit one tree for binary clf.
        super(BinomialDeviance, self).__init__(1)

    def init_estimator(self):
        return LogOddsEstimator()

    def __call__(self, y, pred, sample_weight=None):
        """Compute the negative log-likelihood. """
        # logaddexp(0, v) == log(1.0 + exp(v))
        pred = pred.ravel()
        if sample_weight is None:
            return -np.mean((y * pred) - np.logaddexp(0.0, pred))
        else:
            return (-1./ sample_weight.sum() *
                    np.sum(sample_weight * ((y * pred) - np.logaddexp(0.0, pred))))

    def negative_gradient(self, y, pred, **kargs):
        """Compute the residual (= negative gradient). """
        return y - expit(pred.ravel())

    def hessian(self, y, pred, residual, **kargs):
        """Compute the second derivative """
        return (y - residual) * (1 - y + residual)

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred, sample_weight):
        """Make a single Newton-Raphson step.

        our node estimate is given by:

            sum(w * (y - prob)) / sum(w * prob * (1 - prob))

        we take advantage that: y - prob = residual
        """
        terminal_region = np.where(terminal_regions == leaf)[0]
        residual = residual.take(terminal_region, axis=0)
        y = y.take(terminal_region, axis=0)
        sample_weight = sample_weight.take(terminal_region, axis=0)

        numerator = np.sum(sample_weight * residual)
        denominator = np.sum(sample_weight * (y - residual) * (1 - y + residual))

        # prevents overflow and division by zero
        if abs(denominator) < 1e-150:
            tree.value[leaf, 0, 0] = 0.0
        else:
            tree.value[leaf, 0, 0] = numerator / denominator

    def _score_to_proba(self, score):
        proba = np.ones((score.shape[0], 2), dtype=np.float64)
        proba[:, 1] = expit(score.ravel())
        proba[:, 0] -= proba[:, 1]
        return proba

    def _score_to_decision(self, score):
        proba = self._score_to_proba(score)
        return np.argmax(proba, axis=1)


class MultinomialDeviance(ClassificationLossFunction):
    """Multinomial deviance loss function for multi-class classification.

    For multi-class classification we need to fit ``n_classes`` trees at
    each stage.
    """

    is_multi_class = True

    def __init__(self, n_classes):
        if n_classes < 3:
            raise ValueError("{0:s} requires more than 2 classes.".format(
                self.__class__.__name__))
        super(MultinomialDeviance, self).__init__(n_classes)

    def init_estimator(self):
        return PriorProbabilityEstimator()

    def __call__(self, y, pred, sample_weight=None):
        # create one-hot label encoding
        Y = np.zeros((y.shape[0], self.K), dtype=np.float64)
        for k in range(self.K):
            Y[:, k] = y == k

        if sample_weight is None:
            return np.sum(-1 * (Y * pred).sum(axis=1) +
                          logsumexp(pred, axis=1))
        else:
            return np.sum(-1 * sample_weight * (Y * pred).sum(axis=1) +
                          logsumexp(pred, axis=1))

    def negative_gradient(self, y, pred, k=0, **kwargs):
        """Compute negative gradient for the ``k``-th class. """
        return y - np.nan_to_num(np.exp(pred[:, k] -
                                        logsumexp(pred, axis=1)))

    def hessian(self, y, pred, k=0, **kargs):
        """Compute the second derivative """
        p = np.nan_to_num(np.exp(pred[:, k] - logsumexp(pred, axis=1)))
        return p * (1 - p)

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred, sample_weight):
        """Make a single Newton-Raphson step. """
        terminal_region = np.where(terminal_regions == leaf)[0]
        residual = residual.take(terminal_region, axis=0)
        y = y.take(terminal_region, axis=0)
        sample_weight = sample_weight.take(terminal_region, axis=0)

        numerator = np.sum(sample_weight * residual)

        denominator = np.sum(sample_weight * (y - residual) *
                             (1.0 - y + residual))

        # prevents overflow and division by zero
        if abs(denominator) < 1e-150:
            tree.value[leaf, 0, 0] = 0.0
        else:
            tree.value[leaf, 0, 0] = numerator / denominator

    def _score_to_proba(self, score):
        return np.nan_to_num(
            np.exp(score - (logsumexp(score, axis=1)[:, np.newaxis])))

    def _score_to_decision(self, score):
        proba = self._score_to_proba(score)
        return np.argmax(proba, axis=1)


class ExponentialLoss(ClassificationLossFunction):
    """Exponential loss function for binary classification.

    Same loss as AdaBoost.

    References
    ----------
    Greg Ridgeway, Generalized Boosted Models: A guide to the gbm package, 2007
    """
    def __init__(self, n_classes):
        if n_classes != 2:
            raise ValueError("{0:s} requires 2 classes; got {1:d} class(es)"
                             .format(self.__class__.__name__, n_classes))
        # we only need to fit one tree for binary clf.
        super(ExponentialLoss, self).__init__(1)

    def init_estimator(self):
        return ScaledLogOddsEstimator()

    def __call__(self, y, pred, sample_weight=None):
        pred = pred.ravel()
        if sample_weight is None:
            return np.mean(np.exp(-(2. * y - 1.) * pred))
        else:
            return (1.0 / sample_weight.sum() *
                    np.sum(sample_weight * np.exp(-(2 * y - 1) * pred)))

    def negative_gradient(self, y, pred, **kargs):
        y_ = -(2. * y - 1.)
        return y_ * np.exp(y_ * pred.ravel())

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred, sample_weight):
        terminal_region = np.where(terminal_regions == leaf)[0]
        pred = pred.take(terminal_region, axis=0)
        y = y.take(terminal_region, axis=0)
        sample_weight = sample_weight.take(terminal_region, axis=0)

        y_ = 2. * y - 1.

        numerator = np.sum(y_ * sample_weight * np.exp(-y_ * pred))
        denominator = np.sum(sample_weight * np.exp(-y_ * pred))

        # prevents overflow and division by zero
        if abs(denominator) < 1e-150:
            tree.value[leaf, 0, 0] = 0.0
        else:
            tree.value[leaf, 0, 0] = numerator / denominator

    def _score_to_proba(self, score):
        proba = np.ones((score.shape[0], 2), dtype=np.float64)
        proba[:, 1] = expit(2.0 * score.ravel())
        proba[:, 0] -= proba[:, 1]
        return proba

    def _score_to_decision(self, score):
        return (score.ravel() >= 0.0).astype(np.int)


LOSS_FUNCTIONS = {'ls': LeastSquaresError,
                  'msr': MeanScaleRegressionLoss,
                  'lad': LeastAbsoluteError,
                  'huber': HuberLossFunction,
                  'quantile': QuantileLossFunction,
                  'deviance': None,    # for both, multinomial and binomial
                  'exponential': ExponentialLoss,
                  'tobit': TobitLossFunction,
                  'poisson': PoissonLossFunction,
                  'tweedie': TweedieLossFunction,
                  'gamma': GammaLossFunction
                  }


INIT_ESTIMATORS = {'zero': ZeroEstimator}


BOOSTING_UPDATE = {'gradient', 'hybrid', 'newton'}


BASE_LEARNER = {'tree', 'kernel', 'combined'}


class VerboseReporter(object):
    """Reports verbose output to stdout.

    If ``verbose==1`` output is printed once in a while (when iteration mod
    verbose_mod is zero).; if larger than 1 then output is printed for
    each update.
    """

    def __init__(self, verbose):
        self.verbose = verbose

    def init(self, est, begin_at_stage=0):
        # header fields and line format str
        header_fields = ['Iter', 'Train Loss']
        verbose_fmt = ['{iter:>10d}', '{train_score:>16.4f}']
        # do oob?
        if est.subsample < 1:
            header_fields.append('OOB Improve')
            verbose_fmt.append('{oob_impr:>16.4f}')
        if est.n_iter_no_change is not None:
            header_fields.append('Val Loss')
            verbose_fmt.append('{val_score:>16.4f}')            
        header_fields.append('Remaining Time')
        verbose_fmt.append('{remaining_time:>16s}')

        # print the header line
        print(('%10s ' + '%16s ' *
               (len(header_fields) - 1)) % tuple(header_fields))

        self.verbose_fmt = ' '.join(verbose_fmt)
        # plot verbose info each time i % verbose_mod == 0
        self.verbose_mod = 1
        self.start_time = time()
        self.begin_at_stage = begin_at_stage

    def update(self, j, est):
        """Update reporter with new iteration. """
        do_oob = est.subsample < 1
        # we need to take into account if we fit additional estimators.
        i = j - self.begin_at_stage  # iteration relative to the start iter
        if (i + 1) % self.verbose_mod == 0:
            oob_impr = est.oob_improvement_[j] if do_oob else 0
            if est.n_iter_no_change is not None: 
                val_score = est.val_score_[j]
            else:
                val_score = 0
            remaining_time = ((est.n_estimators - (j + 1)) *
                              (time() - self.start_time) / float(i + 1))
            if remaining_time > 60:
                remaining_time = '{0:.2f}m'.format(remaining_time / 60.0)
            else:
                remaining_time = '{0:.2f}s'.format(remaining_time)
            print(self.verbose_fmt.format(iter=j + 1,
                                          train_score=est.train_score_[j],
                                          val_score=val_score,
                                          oob_impr=oob_impr,
                                          remaining_time=remaining_time))
            if self.verbose == 1 and ((i + 1) // (self.verbose_mod * 10) > 0):
                # adjust verbose frequency (powers of 10)
                self.verbose_mod *= 10

def predict_stages_kernel(estimators_kernel, X, learning_rate, score, pred_kernel_mat=None):
    """Add predictions of ``estimators_kernel[stage]`` to ``score``.

    Each estimator in the stage is scaled by ``learning_rate`` before
    its prediction is added to ``score``.
    """
    
    n_estimators = estimators_kernel.shape[0]
    K = estimators_kernel.shape[1]

    for i in range(n_estimators):
        for k in range(K):
            modi=estimators_kernel[i,k]
            if not pred_kernel_mat is None:
                preds=modi.predict(X=X,pred_kernel_mat=pred_kernel_mat).ravel()
            else:
                preds=modi.predict(X=X).ravel()
            score[:,k]+=learning_rate*preds
            score[:,k][score[:,k]>MAX_VAL_PRED]=MAX_VAL_PRED##Avoid overflow 
            score[:,k][score[:,k]<-MAX_VAL_PRED]=-MAX_VAL_PRED
    return score

def predict_stage_kernel(estimators_kernel, stage, X, learning_rate, score, pred_kernel_mat=None):
    """Add predictions of ``estimators_kernel[stage]`` to ``score``.

    Each estimator in the stage is scaled by ``learning_rate`` before
    its prediction is added to ``score``.
    """
    
    K = estimators_kernel.shape[1]

    for k in range(K):
        modi=estimators_kernel[stage,k]
        if not pred_kernel_mat is None:
            preds=modi.predict(X=X,pred_kernel_mat=pred_kernel_mat).ravel()
        else:
            preds=modi.predict(X=X).ravel()
        score[:,k]+=learning_rate*preds
        score[:,k][score[:,k]>MAX_VAL_PRED]=MAX_VAL_PRED##Avoid overflow 
        score[:,k][score[:,k]<-MAX_VAL_PRED]=-MAX_VAL_PRED
    return score

class BaseBoosting(six.with_metaclass(ABCMeta, BaseEnsemble)):
    """Abstract base class for Boosting. """

    @abstractmethod
    def __init__(self, loss, learning_rate, n_estimators, criterion,
                 min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
                 min_weight_leaf, max_depth, min_impurity_decrease, 
                 init, subsample, max_features, random_state, alpha=0.9, 
                 verbose=0, max_leaf_nodes=None, warm_start=False, 
                 validation_fraction=0.1, 
                 n_iter_no_change=None, tol=1e-4,
                 update_step="hybrid", base_learner="tree", kernel="rbf", scaleX=False, 
                 theta=1, n_neighbors=None, prctg_neighbors=None, range_adjust=1., alphaReg=1.,
                 sparse=False, nystroem=False, n_components=100, 
                 sigma=1., yl=0., yu=1., gamma=1, tweedie_variance_power=1.5):

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.min_weight_leaf=min_weight_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.init = init
        self.random_state = random_state
        self.alpha = alpha
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes
        self.warm_start = warm_start
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.sigma = sigma
        self.yl = yl
        self.yu = yu
        self.gamma = gamma
        self.update_step = update_step
        self.base_learner = base_learner
        self.kernel = kernel
        self.scaleX = scaleX
        self.theta = theta
        self.n_neighbors = n_neighbors
        self.prctg_neighbors = prctg_neighbors
        self.range_adjust = range_adjust
        self.alphaReg = alphaReg
        self.sparse = sparse
        self.nystroem = nystroem
        self.n_components = n_components
        self.tweedie_variance_power = tweedie_variance_power

    def _fit_stage(self, i, X, y, y_pred, sample_weight, sample_mask,
                   random_state, nTreeKernel, X_csc=None, X_csr=None):
        """Fit another stage of ``n_classes_`` trees to the boosting model. """

        assert sample_mask.dtype == np.bool
        loss = self.loss_
        original_y = y

        y_pred_last = y_pred.copy()
        trees=[]
        KernRegrs=[]
        for k in range(loss.K):
            weights = sample_weight.copy()##need to take a copy for the multiclass case with K>1 (otherwise the sample_weights get modified for increasing k...)
            if self.subsample < 1.0:
                # no inplace multiplication!
                weights = weights * sample_mask.astype(np.float64)
            if loss.is_multi_class:
                y = np.array(original_y == k, dtype=np.float64)

            residual = loss.negative_gradient(y, y_pred, k=k,
                                              sample_weight=sample_weight)
            
            if self.update_step=="newton":
                hessian = loss.hessian(y=y, pred=y_pred, residual=residual, 
                                       k=k, sample_weight=sample_weight)
                hessian[hessian < MIN_VAL_HESSIAN] = MIN_VAL_HESSIAN
                weights = weights * hessian
                residual = residual / hessian
                weights = (weights / np.sum(weights) * len(weights))

            """
            Calculate estimators
            """
            if not self.base_learner == "kernel":
                # Calculate regression tree update
                tree = DecisionTreeRegressor(
                    criterion=self.criterion,
                    splitter='best',
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                    min_weight_leaf=self.min_weight_leaf,
                    min_impurity_decrease=self.min_impurity_decrease,
                    max_features=self.max_features,
                    max_leaf_nodes=self.max_leaf_nodes,
                    random_state=random_state,
                    )
    
                if X_csc is not None:
                    tree.fit(X_csc, residual, sample_weight=weights,
                             check_input=False)
                else:
                    tree.fit(X, residual, sample_weight=weights,
                             check_input=False)
    
                # update tree leaves
                if X_csr is not None:
                    loss.update_terminal_regions(tree.tree_, X_csr, y, residual, y_pred,
                                                 weights, sample_mask,
                                                 self.learning_rate, k=k,
                                                 update_step=self.update_step)
                else:
                    loss.update_terminal_regions(tree.tree_, X, y, residual, y_pred,
                                                 weights, sample_mask,
                                                 self.learning_rate, k=k,
                                                 update_step=self.update_step)
            if not self.base_learner == "tree": # XXX ToDo: handling of (i) X_csc and (ii) X_idx_sorted.
                if self.kernel_mat is None:##Initialize kernel matrix once (ToDo: move this out of this function to proper initialization)
                    modi = KernelRidge(alpha=self.alphaReg,theta=self.theta,kernel=self.kernel,
                                             n_neighbors=self.n_neighbors,prctg_neighbors=self.prctg_neighbors,
                                             range_adjust=self.range_adjust,sparse=self.sparse,nystroem=self.nystroem,
                                             n_components=self.n_components)
                    self.kernel_mat=modi._get_kernel(X, nystroem_kernel=self.nystroem)
                    ## 64bit calculations are faster 
                    if not self.kernel_mat.dtype==np.float64: self.kernel_mat=self.kernel_mat.astype(np.float64)
                    if self.theta is None: self.theta=modi.theta
                    if self.nystroem: 
                        self.component_indices = modi.component_indices
                        self.kernel_mat_nystroem_full = modi._get_kernel(X, X[self.component_indices])
                        if not self.kernel_mat_nystroem_full.dtype==np.float64: self.kernel_mat_nystroem_full=self.kernel_mat_nystroem_full.astype(np.float64)
                    if self.update_step=="newton":##No need to precompute inverse of factor of it since it needs to be done in any iteration
                        self.solve_kernel=None
                    else:
                        K=self.kernel_mat.copy()
                        ##Add regularization parameter to kernel matrix
                        if self.sparse:
                            K+=sparse.diags(np.repeat(self.alphaReg, K.shape[0]))
                            self.solve_kernel=sp_linalg.factorized(K)
                        else:
                            K.flat[::K.shape[0] + 1] += self.alphaReg
                            self.solve_kernel=linalg.inv(K)
                modi = KernelRidge(alpha=self.alphaReg,theta=self.theta,kernel_mat=self.kernel_mat,
                                         solve_kernel=self.solve_kernel,kernel=self.kernel,n_neighbors=self.n_neighbors,
                                         prctg_neighbors=self.prctg_neighbors,range_adjust=self.range_adjust,sparse=self.sparse,
                                         nystroem=self.nystroem,n_components=self.n_components,component_indices=self.component_indices)
                if self.update_step=="newton":
                    modi.fit(X, residual, sample_weight=weights)
                else:
                    modi.fit(X, residual)
                if self.nystroem:
                    y_pred_last[:, k] += self.learning_rate * modi.predict(X,pred_kernel_mat=self.kernel_mat_nystroem_full).ravel()
                else:
                    y_pred_last[:, k] += self.learning_rate * modi.predict(X,training_data=True).ravel()
                loss.avoid_overflow(y_pred_last,k)
            if self.base_learner == "tree":
                self.estimators_[nTreeKernel[0], k] = tree
                if k==(loss.K-1): nTreeKernel[0]+=1
            elif self.base_learner == "kernel":
                self.estimators_kernel_[nTreeKernel[1],k]=modi
                y_pred[:, k]=y_pred_last[:, k]
                if k==(loss.K-1): nTreeKernel[1]+=1
            else:
                trees+=[tree]
                KernRegrs+=[modi]  
        """
        Find best base learner
        """
        if not self.base_learner in ["tree","kernel"]:
            lossKernel=loss(original_y,y_pred_last,sample_weight)
            lossTree=loss(original_y,y_pred,sample_weight)
            if lossKernel<lossTree:
                for k in range(loss.K):
                    self.estimators_kernel_[nTreeKernel[1], k]=KernRegrs[k]
                y_pred=y_pred_last
                nTreeKernel[1]+=1
            else:##Select tree
                for k in range(loss.K): self.estimators_[nTreeKernel[0], k] = trees[k]
                nTreeKernel[0]+=1

        return y_pred

    def _check_params(self):
        """Check validity of parameters and raise ValueError if not valid. """
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0 but "
                             "was %r" % self.n_estimators)

        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than 0 but "
                             "was %r" % self.learning_rate)

        if (self.loss not in self._SUPPORTED_LOSS
                or self.loss not in LOSS_FUNCTIONS):
            raise ValueError("Loss '{0:s}' not supported. ".format(self.loss))

        if self.update_step not in BOOSTING_UPDATE:
            raise ValueError("Boosting update step '{0:s}' not supported. ".format(self.update_step))

        if self.base_learner not in BASE_LEARNER:
            raise ValueError("Base learner '{0:s}' not supported. ".format(self.base_learner))

        if ((self.loss in ('huber', 'quantile', 'lad', 'ls')) 
                and (self.update_step == "newton")):
            raise ValueError("Newton updates for loss '{0:s}' not meaningfull "
                             "since Hessian is zero on a non-null set or "
                             "constant. ".format(self.loss))
            
        if ((self.loss in ('exponential')) 
                and (self.update_step == "newton")):
            raise ValueError("Newton updates for loss '{0:s}' currently not supported. ".format(self.loss))
        
        if ((self.loss == 'tweedie') 
                and ((self.tweedie_variance_power <= 1) or (self.tweedie_variance_power>=2))):
            raise ValueError("Tweedie variance power must be in (1,2) but was %r" % self.tweedie_variance_power)
        
        if self.loss == 'deviance':
            loss_class = (MultinomialDeviance
                          if len(self.classes_) > 2
                          else BinomialDeviance)
        else:
            loss_class = LOSS_FUNCTIONS[self.loss]

        if self.loss in ('huber', 'quantile'):
            self.loss_ = loss_class(self.n_classes_, self.alpha)
        elif self.loss in ('tobit'):
            self.loss_ = loss_class(self.n_classes_, self.sigma,
                                    self.yl, self.yu)
        elif self.loss in ('tweedie'):
            self.loss_ = loss_class(self.n_classes_, self.tweedie_variance_power)
        else:
            self.loss_ = loss_class(self.n_classes_)

        if not (0.0 < self.subsample <= 1.0):
            raise ValueError("subsample must be in (0,1] but "
                             "was %r" % self.subsample)

        if self.init is not None:
            if isinstance(self.init, six.string_types):
                if self.init not in INIT_ESTIMATORS:
                    raise ValueError('init="%s" is not supported' % self.init)
            else:
                if (not hasattr(self.init, 'fit')
                        or not hasattr(self.init, 'predict')):
                    raise ValueError("init=%r must be valid BaseEstimator "
                                     "and support both fit and "
                                     "predict" % self.init)

        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0.0, 1.0) but "
                             "was %r" % self.alpha)

        if isinstance(self.max_features, six.string_types):
            if self.max_features == "auto":
                # if is_classification
                if self.n_classes_ > 1:
                    max_features = max(1, int(np.sqrt(self.n_features_)))
                else:
                    # is regression
                    max_features = self.n_features_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError("Invalid value for max_features: %r. "
                                 "Allowed string values are 'auto', 'sqrt' "
                                 "or 'log2'." % self.max_features)
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            if 0. < self.max_features <= 1.:
                max_features = max(int(self.max_features *
                                       self.n_features_), 1)
            else:
                raise ValueError("max_features must be in (0, n_features]")

        self.max_features_ = max_features

        if not isinstance(self.n_iter_no_change,
                          (numbers.Integral, np.integer, type(None))):
            raise ValueError("n_iter_no_change should either be None or an "
                             "integer. %r was passed"
                             % self.n_iter_no_change)
            
        if self.alphaReg < 0:
            raise ValueError("alphaReg must be greater than 0 but "
                             "was %r" % self.alphaReg)
        if self.gamma < 0:
            raise ValueError("gamma must be greater than 0 but "
                             "was %r" % self.gamma)
            
        if ((self.theta is None) & (self.n_neighbors is None) & 
            (self.prctg_neighbors is None) & (self.base_learner in ["kernel","combined"])):
            raise ValueError("At least one from the three parameters theta, "
                             "n_neighbors or prctg_neighbors must be specified")
            
    def _init_state(self):
        """Initialize model state and allocate model state data structures. """

        if self.init is None:
            self.init_ = self.loss_.init_estimator()
        elif isinstance(self.init, six.string_types):
            self.init_ = INIT_ESTIMATORS[self.init]()
        else:
            self.init_ = self.init

        self.estimators_ = np.empty((self.n_estimators, self.loss_.K),
                                    dtype=np.object)
        self.estimators_kernel_ = np.empty((self.n_estimators, self.loss_.K),
                                    dtype=np.object)##Dimension 0: models, 1: reponse variable
        self.kernel_mat = None##Kernel matrix saved in order to construct only once
        self.pred_kernel_mat = None##Same for prediction in staged predictions
        self.component_indices = None##Indices of the data points used for Nystroem sampling
        self.kernel_mat_nystroem_full = None##Kernel matrix needed to make predictions at all training locations when Nystroem sampling is used
        self.scaler=None##Function that scale features
        self.neigh_ind=None##Indices of nearest neighbors for kNN
        self.number_estimators = np.empty((self.n_estimators,2),dtype=np.object)##Number of estimators [0]: trees, [1]: other base learners
        self.train_score_ = np.zeros((self.n_estimators,), dtype=np.float64)
        
        # do oob?
        if self.subsample < 1.0:
            self.oob_improvement_ = np.zeros((self.n_estimators),
                                             dtype=np.float64)
        # do validation?
        if self.n_iter_no_change is not None:
            self.val_score_ = np.zeros((self.n_estimators,), dtype=np.float64)

    def _clear_state(self):
        """Clear the state of the boosting model. """
        if hasattr(self, 'estimators_'):
            self.estimators_ = np.empty((0, 0), dtype=np.object)
        if hasattr(self, 'train_score_'):
            del self.train_score_
        if hasattr(self, 'val_score_'):
            del self.val_score_
        if hasattr(self, 'oob_improvement_'):
            del self.oob_improvement_
        if hasattr(self, 'init_'):
            del self.init_
        if hasattr(self, '_rng'):
            del self._rng

    def _resize_state(self):
        """Add additional ``n_estimators`` entries to all attributes. """
        # self.n_estimators is the number of additional est to fit
        total_n_estimators = self.n_estimators
        if total_n_estimators < self.estimators_.shape[0]:
            raise ValueError('resize with smaller n_estimators %d < %d' %
                             (total_n_estimators, self.estimators_[0]))

        self.estimators_.resize((total_n_estimators, self.loss_.K))
        self.train_score_.resize(total_n_estimators)
        if self.n_iter_no_change is not None:
            self.val_score_.resize(total_n_estimators)
        if (self.subsample < 1 or hasattr(self, 'oob_improvement_')):
            # if do oob resize arrays or create new if not available
            if hasattr(self, 'oob_improvement_'):
                self.oob_improvement_.resize(total_n_estimators)
            else:
                self.oob_improvement_ = np.zeros((total_n_estimators,),
                                                 dtype=np.float64)

    def _is_initialized(self):
        return len(getattr(self, 'estimators_', [])) > 0

    def _check_initialized(self):
        """Check that the estimator is initialized, raising an error if not."""
        check_is_fitted(self, 'estimators_')

    @property
    @deprecated("Attribute n_features was deprecated in version 0.19 and "
                "will be removed in 0.21.")
    def n_features(self):
        return self.n_features_

    def fit(self, X, y, sample_weight=None, monitor=None):
        """Fit the boosting model.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values (strings or integers in classification, real numbers
            in regression)
            For classification, labels must correspond to classes.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        monitor : callable, optional
            The monitor is called after each iteration with the current
            iteration, a reference to the estimator and the local variables of
            ``_fit_stages`` as keyword arguments ``callable(i, self,
            locals())``. If the callable returns ``True`` the fitting procedure
            is stopped. The monitor can be used for various things such as
            computing held-out estimates, early stopping, model introspect, and
            snapshoting.

        Returns
        -------
        self : object
            Returns self.
        """
        # if not warmstart - clear the estimator state
        if not self.warm_start:
            self._clear_state()

        # Check input
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'], dtype=DTYPE)
        
        n_samples, self.n_features_ = X.shape
        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=np.float32)
        else:
            sample_weight = column_or_1d(sample_weight, warn=True)

        check_consistent_length(X, y, sample_weight)

        y = self._validate_y(y, sample_weight)

        if self.n_iter_no_change is not None:
            X, X_val, y, y_val, sample_weight, sample_weight_val = (
                train_test_split(X, y, sample_weight,
                                 random_state=self.random_state,
                                 test_size=self.validation_fraction))
        else:
            X_val = y_val = sample_weight_val = None

        self._check_params()

        if (self.loss == "msr") & (self.min_samples_leaf==1) & (self.update_step in ["gradient", "hybrid"]):
            warnings.warn("Minimum number of samples per leaf should be larger than 1 " 
                          "for mean-scale regression.")

        if ((self.base_learner in ["kernel","combined"]) & 
            (not self.n_neighbors is None)):
            if (len(y)<self.n_neighbors) & (not self.n_neighbors == np.inf):
                raise ValueError("Number of neighbors is larger than number of samples.")

        if not self._is_initialized():
            if self.scaleX:
                self.scaler = preprocessing.StandardScaler().fit(X)
                X = self.scaler.transform(X)
            
            # init state
            self._init_state()

            # fit initial model - FIXME make sample_weight optional
            self.init_.fit(X, y, sample_weight)

            # init predictions
            y_pred = self.init_.predict(X)
            begin_at_stage = 0

            # The rng state must be preserved if warm_start is True
            self._rng = check_random_state(self.random_state)

        else:
            # add more estimators to fitted model
            # invariant: warm_start = True
            if self.n_estimators < self.estimators_.shape[0]:
                raise ValueError('n_estimators=%d must be larger or equal to '
                                 'estimators_.shape[0]=%d when '
                                 'warm_start==True'
                                 % (self.n_estimators,
                                    self.estimators_.shape[0]))
            begin_at_stage = self.estimators_.shape[0]
            # The requirements of _decision_function (called in two lines
            # below) are more constrained than fit. It accepts only CSR
            # matrices.
            X = check_array(X, dtype=DTYPE, order="C", accept_sparse='csr')
            y_pred = self._decision_function(X)
            self._resize_state()

        # fit the boosting stages
        # XXX ToDo: should n_stages be deleted? (redundant due to self.number_estimators)
        n_stages = self._fit_stages(X, y, y_pred, sample_weight, self._rng,
                                    X_val, y_val, sample_weight_val,
                                    begin_at_stage, monitor)
        
        n_trees = self.number_estimators[n_stages-1][0]
        n_kernel = self.number_estimators[n_stages-1][1]
        if not self.base_learner in ["tree","kernel"]:
            print("Number of trees="+str(n_trees)+", number of kernel functions="+str(n_kernel))
        # change shape of arrays after fit (early-stopping or additional iterations)
        if n_trees != self.estimators_.shape[0]:
            self.estimators_ = self.estimators_[:n_trees]
        if n_kernel != self.estimators_kernel_.shape[0]:
            self.estimators_kernel_ = self.estimators_kernel_[:n_kernel]
            
        if n_stages != (n_trees+n_kernel):
            self.train_score_ = self.train_score_[:n_stages]
            self.val_score_ = self.val_score_[:n_stages]
            if hasattr(self, 'oob_improvement_'):
                self.oob_improvement_ = self.oob_improvement_[:n_stages]

        self.n_estimators_ = n_stages
        return self

    def _fit_stages(self, X, y, y_pred, sample_weight, random_state,
                    X_val, y_val, sample_weight_val,
                    begin_at_stage=0, monitor=None):
        """Iteratively fits the stages.

        For each stage it computes the progress (OOB, train score)
        and delegates to ``_fit_stage``.
        Returns the number of stages fit; might differ from ``n_estimators``
        due to early stopping.
        """
        n_samples = X.shape[0]
        do_oob = self.subsample < 1.0
        sample_mask = np.ones((n_samples, ), dtype=np.bool)
        n_inbag = max(1, int(self.subsample * n_samples))
        loss_ = self.loss_

        if self.verbose:
            verbose_reporter = VerboseReporter(self.verbose)
            verbose_reporter.init(self, begin_at_stage)

        X_csc = csc_matrix(X) if issparse(X) else None
        X_csr = csr_matrix(X) if issparse(X) else None

        if self.n_iter_no_change is not None:
            loss_history = np.ones(self.n_iter_no_change) * np.inf
            # We create a generator to get the predictions for X_val after
            # the addition of each successive stage
            y_val_pred_iter = self._staged_decision_function(X_val)

        # perform boosting iterations
        i = begin_at_stage
        nTreeKernel=[0,0]##Number of trees [0] and kernel regressors [1] at iteration i
        
        n_iter_no_improve = 0
        best_val_score = np.inf
        best_iter = 0
        
        for i in range(begin_at_stage, self.n_estimators):

            # subsampling
            if do_oob:
                sample_mask = _random_sample_mask(n_samples, n_inbag,
                                                  random_state)
                # OOB score before adding this stage
                old_oob_score = loss_(y[~sample_mask],
                                      y_pred[~sample_mask],
                                      sample_weight[~sample_mask])

            # fit next stage of trees
            y_pred = self._fit_stage(i, X, y, y_pred, sample_weight,
                                     sample_mask, random_state,
                                     nTreeKernel, X_csc, X_csr)
            self.number_estimators[i]=nTreeKernel

            # track deviance (= loss)
            if do_oob:
                self.train_score_[i] = loss_(y[sample_mask],
                                             y_pred[sample_mask],
                                             sample_weight[sample_mask])
                self.oob_improvement_[i] = (
                    old_oob_score - loss_(y[~sample_mask],
                                          y_pred[~sample_mask],
                                          sample_weight[~sample_mask]))
            else:
                # no need to fancy index w/ no subsampling
                self.train_score_[i] = loss_(y, y_pred, sample_weight)
            
            # Calculate validation score
            if self.n_iter_no_change is not None:
                # By calling next(y_val_pred_iter), we get the predictions
                # for X_val after the addition of the current stage
                self.val_score_[i] = loss_(y_val, next(y_val_pred_iter),
                                        sample_weight_val)
                
            if self.verbose > 0:
                verbose_reporter.update(i, self)

            if monitor is not None:
                early_stopping = monitor(i, self, locals())
                if early_stopping:
                    break

            # We also provide an early stopping based on the score from
            # validation set (X_val, y_val), if n_iter_no_change is set
            if self.n_iter_no_change is not None:
                # # Require validation_score to be better (less) than at least
                # # one of the last n_iter_no_change evaluations
                # if np.any(self.val_score_[i] + self.tol < loss_history):
                #     loss_history[i % len(loss_history)] = self.val_score_[i]
                # else:
                #     break
                
                # Early stop if consecutive (n_iter_no_change) iterations have not
                # made any improvement in val score
                if i == 0:
                    best_val_score = self.val_score_[i]
                else:
                    if self.val_score_[i] < (best_val_score + self.tol):
                        best_iter = i
                        best_val_score = self.val_score_[i]
                        n_iter_no_improve = 0
                    else:
                        n_iter_no_improve += 1
                if n_iter_no_improve == self.n_iter_no_change:
                    break
        
        if self.n_iter_no_change is not None:
            return best_iter + 1
        else:
            return i + 1

    def _make_estimator(self, append=True):
        # we don't need _make_estimator
        raise NotImplementedError()

    def _init_decision_function(self, X):
        """Check input and compute prediction of ``init``. """
        self._check_initialized()
        # number of estimators is initialized as np.empty array
        n_trees = 0
        if self.number_estimators[self.n_estimators-1][0] is not None:
            n_trees = self.number_estimators[self.n_estimators-1][0]
        # XXX ToDo make sure that '_validate_X_predict' is also performed when no tree is present
        if n_trees>0: X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)
        if X.shape[1] != self.n_features_:
            raise ValueError("X.shape[1] should be {0:d}, not {1:d}.".format(
                self.n_features_, X.shape[1]))
        score = self.init_.predict(X).astype(np.float64)
        return score

    def _decision_function(self, X):
        # for use in inner loop, not raveling the output in single-class case,
        # not doing input validation.
        # XXX ToDo: unify treatment of 'self.pred_kernel_mat is None' in here and _staged_decision_function
        if self.scaleX: X = self.scaler.transform(X)
        score = self._init_decision_function(X)
        predict_stages(self.estimators_, X, self.learning_rate, score)
        if self.estimators_kernel_.shape[0]>0:
            modi=self.estimators_kernel_[0,0]
            modi.theta=self.theta
            pred_kernel_mat = modi._get_kernel(X, modi.X_fit_)
            if not pred_kernel_mat.dtype==np.float64: pred_kernel_mat=pred_kernel_mat.astype(np.float64)
            predict_stages_kernel(self.estimators_kernel_, X, self.learning_rate, score, pred_kernel_mat)
        return score

    def _staged_decision_function(self, X):
        """Compute decision function of ``X`` for each iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        score : generator of array, shape = [n_samples, k]
            The decision function of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
            Regression and binary classification are special cases with
            ``k == 1``, otherwise ``k==n_classes``.
        """
        X = check_array(X, dtype=DTYPE, order="C",  accept_sparse='csr')
        score = self._init_decision_function(X)
        pred_kernel_mat = None
        for i in range(self.number_estimators.shape[0]):
            # Skip if there is no estimator
            if self.number_estimators[i][0] is None:
                continue
            
            if (i == 0):
                if self.number_estimators[0][0]==1:
                    predict_stage(self.estimators_, self.number_estimators[i][0]-1, X, self.learning_rate, score)
                elif (pred_kernel_mat is None) & (self.estimators_kernel_.shape[0]>0):
                        modi=self.estimators_kernel_[0,0]
                        modi.theta=self.theta
                        pred_kernel_mat = modi._get_kernel(X, modi.X_fit_)
                        if not pred_kernel_mat.dtype==np.float64: pred_kernel_mat=pred_kernel_mat.astype(np.float64)
                        predict_stage_kernel(self.estimators_kernel_, self.number_estimators[i][1]-1, X, self.learning_rate, score, pred_kernel_mat)
            elif (i > 0) & (self.number_estimators[i][0]>self.number_estimators[i-1][0]): # tree is used for prediction
                predict_stage(self.estimators_, self.number_estimators[i][0]-1, X, self.learning_rate, score)
            else: # kernel machine is used for prediction
                if (pred_kernel_mat is None) & (self.estimators_kernel_.shape[0]>0):
                    modi=self.estimators_kernel_[0,0]
                    modi.theta=self.theta
                    pred_kernel_mat = modi._get_kernel(X, modi.X_fit_)
                    if not pred_kernel_mat.dtype==np.float64: pred_kernel_mat=pred_kernel_mat.astype(np.float64)
                predict_stage_kernel(self.estimators_kernel_, self.number_estimators[i][1]-1, X, self.learning_rate, score, pred_kernel_mat)
            yield score.copy()

    @property
    def feature_importances_(self):
        """Return the feature importances (the higher, the more important the
           feature).

        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        """
        
        if not self.base_learner == "tree":
            raise ValueError("Feature importances are only "
                             "defined for trees as base "
                             "learners. Use option 'base_learner=\"tree\"'.")
        
        self._check_initialized()

        total_sum = np.zeros((self.n_features_, ), dtype=np.float64)
        for stage in self.estimators_:
            stage_sum = sum(tree.feature_importances_
                            for tree in stage) / len(stage)
            total_sum += stage_sum

        importances = total_sum / len(self.estimators_)
        return importances

    def _validate_y(self, y, sample_weight):
        # 'sample_weight' is not utilised but is used for
        # consistency with similar method _validate_y of GBC in scikit-learn
        self.n_classes_ = 1
        if y.dtype.kind == 'O':
            y = y.astype(np.float64)
        # Default implementation
        return y

    def apply(self, X):
        """Apply trees in the ensemble to X, return leaf indices.

        .. versionadded:: 0.17

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will
            be converted to a sparse ``csr_matrix``.

        Returns
        -------
        X_leaves : array_like, shape = [n_samples, n_estimators, n_classes]
            For each datapoint x in X and for each tree in the ensemble,
            return the index of the leaf x ends up in each estimator.
            In the case of binary classification n_classes is 1.
        """

        self._check_initialized()
        X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)

        # n_classes will be equal to 1 in the binary classification or the
        # regression case.
        n_estimators, n_classes = self.estimators_.shape
        leaves = np.zeros((X.shape[0], n_estimators, n_classes))

        for i in range(n_estimators):
            for j in range(n_classes):
                estimator = self.estimators_[i, j]
                leaves[:, i, j] = estimator.apply(X, check_input=False)

        return leaves


class BoostingClassifier(BaseBoosting, ClassifierMixin):
    """Boosting for classification.

    Parameters
    ----------
    loss : {'deviance', 'exponential'}, optional (default='deviance')
        loss function to be optimized. 'deviance' refers to the logistic 
        regression loss for binary classification, and the cross-entropy loss
        with the softmax function for multiclass classification.
    
    update_step : string, default="hybrid"
        Defines how boosting updates are calculated. Use either "gradient" for gradient boosting
        or "newton" for Newton boosting (if applicable). "hybrid" uses a gradient step for finding the structure
        of trees and a Newton step for finding the leaf values. For kernel boosting, "hybrid" uses
        gradient descent.

    base_learner : string, default="tree"
        Base learners used in boosting updates. Choose among "tree" for trees, 
        "kernel" for reproducing kernel Hilbert space (RKHS) regression
        functions, and "combined" for a combination of the two.

    learning_rate : float, optional (default=0.1)
        learning rate shrinks the contribution of each base learner by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.

    n_estimators : int (default=100)
        The number of boosting stages to perform.

    max_depth : integer, optional (default=5)
        Maximum depth of the regression trees. The maximum
        depth limits the number of nodes in the tree. This value determines 
        the interaction of the predictor variables.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_leaf : float, optional (default=1.)
        The minimum number of weighted samples required to be at a leaf node.
        If Newton boosting is used, this corresponds to the equivalent (i.e.,
        normalized) number of weighted samples where the weights are determined
        based on the second derivatives / Hessians.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "friedman_mse" for the mean squared error with improvement
        score by Friedman, "mse" for mean squared error, and "mae" for
        the mean absolute error.

    subsample : float, optional (default=1.0)
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. `subsample` interacts with the parameter `n_estimators`.
        Choosing `subsample < 1.0` leads to a reduction of variance
        and an increase in bias.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Choosing `max_features < n_features` leads to a reduction of variance
        and an increase in bias.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    init : BaseEstimator, None, optional (default=None)
        An estimator object that is used to compute the initial
        predictions. ``init`` has to provide ``fit`` and ``predict``.
        If None it uses ``loss.init_estimator``.

    verbose : int, default: 0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more trees the lower the frequency). If greater
        than 1 then it prints progress and performance for every tree.

    warm_start : bool, default: False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just erase the
        previous solution.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    validation_fraction : float, optional, default 0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if ``n_iter_no_change`` is set to an integer.

    n_iter_no_change : int, default None
        ``n_iter_no_change`` is used to decide if early stopping will be used
        to terminate training when validation score is not improving. By
        default it is set to None to disable early stopping. If set to a
        number, it will set aside ``validation_fraction`` size of the training
        data as validation and terminate training when validation score is not
        improving in all of the previous ``n_iter_no_change`` numbers of
        iterations.

    tol : float, optional, default 1e-4
        Tolerance for the early stopping. When the loss is not improving
        by at least tol for ``n_iter_no_change`` iterations (if set to a
        number), the training stops.

    kernel : string, default="rbf"
        Kernel function used for kernel boosting. Currently, supports "laplace", "rbf", and "GW" 
        (generalied Wendland with "smoothness parameter" mu=1).

    scaleX : bool, default: False
        When set to ``True``, features are scaled to mean zero and variance one.

    theta : float, default: 1.
        Range parameter of the kernel functions which determines how fast the kernel function
        decays with distance.

    n_neighbors : int, default: None
        If the range parameter 'theta' is not given, it can be determined from the data using this
        parameter. The parameter 'theta' is chosen as the average distance of the 'n_neighbors' 
        nearest neighbors distances. The parameter 'range_adjust' can be used to modify this. 
        If range_adjust=3 or range_adjust=4.6, 'theta' is chosen such that the kernel function has 
        decayed to essentially zero (0.05 or 0.01, respectively) at the average distance of the 
        'n_neighbors' nearest neighbors (for rbf and laplace kernel).

    prctg_neighbors : float, default: None
        Alternative way of specifying the number of nearest neighbors 'n_neighbors'.
        If n_neighbors=None, it is set to prctg_neighbors*n_samples where n_samples denotes the 
        number of training samples.

    range_adjust : float, default: 1.
        See documentation on 'n_neighbors'.

    alphaReg : float, default: 1.
        Regularization parameter for kernel Ridge regression boosting updates. This is added to the diagonal of the
        kernel matrix. Must be a non-negative number. A non-zero value helps to avoid singular matrices.

    sparse : bool, default: False
        When set to ``True``, sparse matrices are used (only meaningfull for kernel="GW").
        
    nystroem : boolean, default=None
        Indicates whether Nystroem sampling is used or not for kernel boosting.

    n_components : int, detault = 100
        Number of data points used in Nystroem sampling for kernel boosting.

    Attributes
    ----------
    n_estimators_ : int
        The number of estimators as selected by early stopping (if
        ``n_iter_no_change`` is specified). Otherwise it is set to
        ``n_estimators``.

        .. versionadded:: 0.20

    feature_importances_ : array, shape = [n_features]
        The feature importances (the higher, the more important the feature).

    oob_improvement_ : array, shape = [n_estimators]
        The improvement in loss (= deviance) on the out-of-bag samples
        relative to the previous iteration.
        ``oob_improvement_[0]`` is the improvement in
        loss of the first stage over the ``init`` estimator.

    train_score_ : array, shape = [n_estimators]
        The i-th score ``train_score_[i]`` is the deviance (= loss) of the
        model at iteration ``i`` on the in-bag sample.
        If ``subsample == 1`` this is the deviance on the training data.

    loss_ : LossFunction
        The concrete ``LossFunction`` object.

    init_ : BaseEstimator
        The estimator that provides the initial predictions.
        Set via the ``init`` argument or ``loss.init_estimator``.

    estimators_ : ndarray of DecisionTreeRegressor, shape = [n_estimators, ``loss_.K``]
        The collection of fitted sub-estimators. ``loss_.K`` is 1 for binary
        classification, otherwise n_classes.

    Notes
    -----
    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.

    References
    ----------
    Friedman, J. H. (2001). Greedy function approximation: a gradient boosting
    machine. Annals of statistics, 1189-1232.
    
    Sigrist, F., & Hirnschall, C. (2017). Grabit: Gradient Tree Boosted Tobit
    Models for Default Prediction. arXiv preprint arXiv:1711.08695.
    
    Sigrist, F. (2018). Gradient and Newton Boosting for Classification and
    Regression. arXiv preprint arXiv:1808.03064.
    
    Sigrist, F. (2019). KTBoost: Combined Kernel and Tree Boosting. arXiv 
    preprint arXiv:1902.03999.
    """

    _SUPPORTED_LOSS = ('deviance', 'exponential')

    def __init__(self, loss='deviance', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion='mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 min_weight_leaf=1., max_depth=5, min_impurity_decrease=0.,
                 init=None, random_state=None, max_features=None, verbose=0,
                 max_leaf_nodes=None, warm_start=False,
                 validation_fraction=0.1,
                 n_iter_no_change=None, tol=1e-4, update_step="hybrid",
                 base_learner="tree", kernel="rbf", scaleX=False, theta=1, 
                 n_neighbors=None, prctg_neighbors=None, range_adjust=1., alphaReg=1.,
                 sparse=False, nystroem=False, n_components=100):

        super(BoostingClassifier, self).__init__(
            loss=loss, learning_rate=learning_rate, n_estimators=n_estimators, criterion=criterion,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf, min_weight_leaf=min_weight_leaf,
            max_depth=max_depth, init=init, subsample=subsample, max_features=max_features,
            random_state=random_state, verbose=verbose, max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease, warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, tol=tol, update_step=update_step, 
            base_learner=base_learner, kernel=kernel, scaleX=scaleX, theta=theta, 
            n_neighbors=n_neighbors, prctg_neighbors=prctg_neighbors, range_adjust=range_adjust,
            alphaReg=alphaReg, sparse=sparse, nystroem=nystroem, n_components=n_components)

    def _validate_y(self, y, sample_weight):
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        n_trim_classes = np.count_nonzero(np.bincount(y, sample_weight))
        if n_trim_classes < 2:
            raise ValueError("y contains %d class after sample_weight "
                             "trimmed classes with zero weights, while a "
                             "minimum of 2 classes are required."
                             % n_trim_classes)
        self.n_classes_ = len(self.classes_)
        return y
    

    def decision_function(self, X):
        """Compute the decision function of ``X``.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        score : array, shape = [n_samples, n_classes] or [n_samples]
            The decision function of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
            Regression and binary classification produce an array of shape
            [n_samples].
        """
        X = check_array(X, dtype=DTYPE, order="C",  accept_sparse='csr')
        score = self._decision_function(X)
        if score.shape[1] == 1:
            return score.ravel()
        return score

    def staged_decision_function(self, X):
        """Compute decision function of ``X`` for each iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        score : generator of array, shape = [n_samples, k]
            The decision function of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
            Regression and binary classification are special cases with
            ``k == 1``, otherwise ``k==n_classes``.
        """
        for dec in self._staged_decision_function(X):
            # no yield from in Python2.X
            yield dec

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted values.
        """
        score = self.decision_function(X)
        decisions = self.loss_._score_to_decision(score)
        return self.classes_.take(decisions, axis=0)

    def staged_predict(self, X):
        """Predict class at each stage for X.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : generator of array of shape = [n_samples]
            The predicted value of the input samples.
        """
        for score in self._staged_decision_function(X):
            decisions = self.loss_._score_to_decision(score)
            yield self.classes_.take(decisions, axis=0)

    def predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Raises
        ------
        AttributeError
            If the ``loss`` does not support probabilities.

        Returns
        -------
        p : array of shape = [n_samples]
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        score = self.decision_function(X)
        try:
            return self.loss_._score_to_proba(score)
        except NotFittedError:
            raise
        except AttributeError:
            raise AttributeError('loss=%r does not support predict_proba' %
                                 self.loss)

    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Raises
        ------
        AttributeError
            If the ``loss`` does not support probabilities.

        Returns
        -------
        p : array of shape = [n_samples]
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        proba = self.predict_proba(X)
        return np.log(proba)

    def staged_predict_proba(self, X):
        """Predict class probabilities at each stage for X.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : generator of array of shape = [n_samples]
            The predicted value of the input samples.
        """
        try:
            for score in self._staged_decision_function(X):
                yield self.loss_._score_to_proba(score)
        except NotFittedError:
            raise
        except AttributeError:
            raise AttributeError('loss=%r does not support predict_proba' %
                                 self.loss)


class BoostingRegressor(BaseBoosting, RegressorMixin):
    """Boosting for regression.

    Parameters
    ----------
    loss : {'ls', 'lad', 'huber', 'quantile', 'poisson', 'tweedie', 'gamma', 
            'tobit', 'msr'}, optional (default='ls')
        loss function to be optimized. 'ls' refers to squared loss.
        'lad' (least absolute deviation) is a highly robust
        loss function solely based on order information of the input
        variables. 'huber' is a combination of the two. 'quantile'
        allows quantile regression (use `alpha` to specify the quantile).
        'tobit' corresponds to a Tobit loss. 'msr' is a linear regression model
        where both the mean and the logarithm of the standard deviation are
        varying.
        
    update_step : string, default="hybrid"
        Defines how boosting updates are calculated. Use either "gradient" for gradient boosting
        or "newton" for Newton boosting (if applicable). "hybrid" uses a gradient step for finding the structure
        of trees and a Newton step for finding the leaf values. For kernel boosting, "hybrid" uses
        gradient descent.

    base_learner : string, default="tree"
        Base learners used in boosting updates. Choose among "tree" for trees, 
        "kernel" for reproducing kernel Hilbert space (RKHS) regression
        functions, and "combined" for a combination of the two.

    learning_rate : float, optional (default=0.1)
        learning rate shrinks the contribution of each base learner by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.

    n_estimators : int (default=100)
        The number of boosting stages to perform.

    max_depth : integer, optional (default=5)
        Maximum depth of the regression trees. The maximum
        depth limits the number of nodes in the tree. This value determines 
        the interaction of the predictor variables.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_leaf : float, optional (default=1.)
        The minimum number of weighted samples required to be at a leaf node.
        If Newton boosting is used, this corresponds to the equivalent (i.e.,
        normalized) number of weighted samples where the weights are determined
        based on the second derivatives / Hessians.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "friedman_mse" for the mean squared error with improvement
        score by Friedman, "mse" for mean squared error, and "mae" for
        the mean absolute error.

    subsample : float, optional (default=1.0)
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic
        Boosting. `subsample` interacts with the parameter `n_estimators`.
        Choosing `subsample < 1.0` leads to a reduction of variance
        and an increase in bias.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Choosing `max_features < n_features` leads to a reduction of variance
        and an increase in bias.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    alpha : float (default=0.9)
        The alpha-quantile of the huber loss function and the quantile
        loss function. Only if ``loss='huber'`` or ``loss='quantile'``.

    init : BaseEstimator, None, optional (default=None)
        An estimator object that is used to compute the initial
        predictions. ``init`` has to provide ``fit`` and ``predict``.
        If None it uses ``loss.init_estimator``.

    verbose : int, default: 0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more trees the lower the frequency). If greater
        than 1 then it prints progress and performance for every tree.

    warm_start : bool, default: False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just erase the
        previous solution.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    validation_fraction : float, optional, default 0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if early_stopping is True

    n_iter_no_change : int, default None
        ``n_iter_no_change`` is used to decide if early stopping will be used
        to terminate training when validation score is not improving. By
        default it is set to None to disable early stopping. If set to a
        number, it will set aside ``validation_fraction`` size of the training
        data as validation and terminate training when validation score is not
        improving in all of the previous ``n_iter_no_change`` numbers of
        iterations.

    tol : float, optional, default 1e-4
        Tolerance for the early stopping. When the loss is not improving
        by at least tol for ``n_iter_no_change`` iterations (if set to a
        number), the training stops.

    kernel : string, default="rbf"
        Kernel function used for kernel boosting. Currently, supports "laplace", "rbf", and "GW" 
        (generalied Wendland with "smoothness parameter" mu=1).

    scaleX : bool, default: False
        When set to ``True``, features are scaled to mean zero and variance one.

    theta : float, default: 1.
        Range parameter of the kernel functions which determines how fast the kernel function
        decays with distance.

    n_neighbors : int, default: None
        If the range parameter 'theta' is not given, it can be determined from the data using this
        parameter. The parameter 'theta' is chosen as the average distance of the 'n_neighbors' 
        nearest neighbors distances. The parameter 'range_adjust' can be used to modify this. 
        If range_adjust=3 or range_adjust=4.6, 'theta' is chosen such that the kernel function has 
        decayed to essentially zero (0.05 or 0.01, respectively) at the average distance of the 
        'n_neighbors' nearest neighbors (for rbf and laplace kernel).

    prctg_neighbors : float, default: None
        Alternative way of specifying the number of nearest neighbors 'n_neighbors'.
        If n_neighbors=None, it is set to prctg_neighbors*n_samples where n_samples denotes the 
        number of training samples.

    range_adjust : float, default: 1.
        See documentation on 'n_neighbors'.

    alphaReg : float, default: 1.
        Regularization parameter for kernel Ridge regression boosting updates. This is added to the diagonal of the
        kernel matrix. Must be a non-negative number. A non-zero value helps to avoid singular matrices.

    sparse : bool, default: False
        When set to ``True``, sparse matrices are used (only meaningfull for kernel="GW").
        
    nystroem : boolean, default=None
        Indicates whether Nystroem sampling is used or not for kernel boosting.

    n_components : int, detault = 100
        Number of data points used in Nystroem sampling for kernel boosting.
        
    sigma : float, optional, default=1.
        Standard deviation of the latent variable in a Tobit model.

    yl : float, optional, default=0.
        Lower limit of the Tobit model. If there is no lower censoring,
        simply set this parameter to a low value (lower than all data points).

    yu : float, optional, default=1.
        Upper limit of the Tobit model. If there is no upper censoring,
        simply set this parameter to a high value (higher than all data points).

    gamma : float, default=1.
        Shape parameter for gamma regression.

    tweedie_variance_power: float, default=1.5
        Parameter for tweedie loss.
        
    Attributes
    ----------
    feature_importances_ : array, shape = [n_features]
        The feature importances (the higher, the more important the feature).

    oob_improvement_ : array, shape = [n_estimators]
        The improvement in loss (= deviance) on the out-of-bag samples
        relative to the previous iteration.
        ``oob_improvement_[0]`` is the improvement in
        loss of the first stage over the ``init`` estimator.

    train_score_ : array, shape = [n_estimators]
        The i-th score ``train_score_[i]`` is the deviance (= loss) of the
        model at iteration ``i`` on the in-bag sample.
        If ``subsample == 1`` this is the deviance on the training data.

    loss_ : LossFunction
        The concrete ``LossFunction`` object.

    init_ : BaseEstimator
        The estimator that provides the initial predictions.
        Set via the ``init`` argument or ``loss.init_estimator``.

    estimators_ : ndarray of DecisionTreeRegressor, shape = [n_estimators, 1]
        The collection of fitted sub-estimators.

    Notes
    -----
    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.

    References
    ----------
    Friedman, J. H. (2001). Greedy function approximation: a gradient boosting
    machine. Annals of statistics, 1189-1232.
    
    Sigrist, F., & Hirnschall, C. (2017). Grabit: Gradient Tree Boosted Tobit
    Models for Default Prediction. arXiv preprint arXiv:1711.08695.
    
    Sigrist, F. (2018). Gradient and Newton Boosting for Classification and
    Regression. arXiv preprint arXiv:1808.03064.
    
    Sigrist, F. (2019). KTBoost: Combined Kernel and Tree Boosting. arXiv 
    preprint arXiv:1902.03999.
    """

    _SUPPORTED_LOSS = ('ls', 'lad', 'huber', 'quantile', 'tobit', 'poisson',
                       'tweedie', 'gamma', 'msr')

    def __init__(self, loss='ls', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion='mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 min_weight_leaf=1., max_depth=5, min_impurity_decrease=0.,
                 init=None, random_state=None,
                 max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None,
                 warm_start=False, validation_fraction=0.1,
                 n_iter_no_change=None, tol=1e-4,
                 update_step="hybrid", base_learner="tree", kernel="rbf", scaleX=False, 
                 theta=1, n_neighbors=None, prctg_neighbors=None, range_adjust=1., alphaReg=1.,
                 sparse=False, nystroem=False, n_components=100,
                 sigma=1., yl=0., yu=1., gamma=1, tweedie_variance_power=1.5):

        super(BoostingRegressor, self).__init__(
            loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
            criterion=criterion, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf,
            min_weight_leaf=min_weight_leaf, max_depth=max_depth, init=init, subsample=subsample,
            max_features=max_features, min_impurity_decrease=min_impurity_decrease,
            random_state=random_state, alpha=alpha, 
            verbose=verbose, max_leaf_nodes=max_leaf_nodes, warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, tol=tol, sigma=sigma,
            yl=yl, yu=yu, gamma=gamma, update_step=update_step, base_learner=base_learner, 
            kernel=kernel, scaleX=scaleX, theta=theta,  n_neighbors=n_neighbors, 
            prctg_neighbors=prctg_neighbors, range_adjust=range_adjust, alphaReg=alphaReg, 
            sparse=sparse, nystroem=nystroem, n_components=n_components, tweedie_variance_power=tweedie_variance_power)
        
    def _validate_y(self, y, sample_weight):
        self.n_classes_ = 1
        if y.dtype.kind == 'O':
            y = y.astype(np.float64)
        if self.loss in ('tweedie', 'poisson', 'gamma'):
            if np.min(y) < 0:
                raise ValueError("y cannot be smaller than 0 for the '{0:s}' loss. ".format(self.loss))
        if self.loss == 'tobit':
            if np.min(y) < self.yl or np.max(y) > self.yu:
                warnings.warn("Found y value outside the interval [%r,%r]  " % (self.yl,self.yu))
        return y
        
    def predict(self, X):
        """Predict regression target for X.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted values.
        """
        X = check_array(X, dtype=DTYPE, order="C",  accept_sparse='csr')
        pred = self._decision_function(X)
        if self.loss_.K == 1:
            if self.loss in ('tweedie', 'poisson', 'gamma'):
                return np.exp(pred.ravel())
            return pred.ravel()
        else:
            if self.loss in ('tweedie', 'poisson', 'gamma'):
                return np.exp(pred)
            return pred

    def staged_predict(self, X):
        """Predict regression target at each stage for X.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : generator of array of shape = [n_samples]
            The predicted value of the input samples.
        """
        for y in self._staged_decision_function(X):
            if self.loss_.K == 1:
                if self.loss == 'tweedie':
                    yield np.exp(y)
                yield y.ravel()
            else:
                if self.loss == 'tweedie':
                    yield np.exp(y)
                yield y

    def apply(self, X):
        """Apply trees in the ensemble to X, return leaf indices.

        .. versionadded:: 0.17

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will
            be converted to a sparse ``csr_matrix``.

        Returns
        -------
        X_leaves : array_like, shape = [n_samples, n_estimators]
            For each datapoint x in X and for each tree in the ensemble,
            return the index of the leaf x ends up in each estimator.
        """

        leaves = super(BoostingRegressor, self).apply(X)
        leaves = leaves.reshape(X.shape[0], self.estimators_.shape[0])
        return leaves

def plot_feature_importances(model,feature_names=None,maxFeat=None,
                             title='Feature importances',xlab='Features', 
                             ax=None, **fig_kw):
    """Creates a plot with feature importances calculated as described in 
    Friedman (2001)

    Parameters
    ----------
    model : BaseBoosting
        A fitted boosting model.
    feature_names : seq of str or one dimensional numpy array, default None
        Name of each feature; feature_names[i] holds
        the name of the feature with index i.
    maxFeat : int, default None
        The maximal number of features to be plotted
    ax : Matplotlib axis object, default None
        An axis object onto which the plots will be drawn.
    **fig_kw : dict
        Dict with keywords passed to the figure() call.
        Note that all keywords not recognized above will be automatically
        included here.
        
    Examples
    --------
    >>> Xtrain=np.random.rand(1000,10)
    >>> ytrain=2*Xtrain[:,0]+2*Xtrain[:,1]+np.random.rand(1000)
    >>> model = KTBoost.BoostingRegressor()
    >>> model.fit(Xtrain,ytrain)
    >>> feat_imp = model.feature_importances_ ## Extract feature importances
    >>> ## Alternatively, plot feature importances directly
    >>> KTBoost.plot_feature_importances(model=model,
                                         feature_names=feature_names,
                                         maxFeat=10)
    """
    if not isinstance(model, BaseBoosting):
        raise ValueError('model has to be an instance of BaseBoosting')
    check_is_fitted(model, 'estimators_')
    feature_importances=model.feature_importances_
    if feature_names is None: feature_names=np.array(range(0,len(feature_importances)))
    if not isinstance(feature_names, np.ndarray): feature_names = np.array(feature_names)
    order = feature_importances.argsort()[::-1][:len(feature_importances)]
    if not maxFeat is None: order = order[:maxFeat]
    feature_importances = feature_importances[order]
    feature_names = feature_names[order]
    
    if ax is None:
        fig = plt.figure(**fig_kw)
    else:
        fig = ax.get_figure()
        fig.clear()
        
    auxindex = np.arange(len(feature_importances))
    plt.bar(auxindex, feature_importances, color='black', alpha=0.5)
    plt.xlabel(xlab, fontsize=18)
    plt.ylabel('Importance', fontsize=18)
    plt.title(title, fontsize=20)
    plt.xticks(auxindex, feature_names,rotation=90, fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()