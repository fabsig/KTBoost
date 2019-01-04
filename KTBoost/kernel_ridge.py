"""This module implements kernel ridge regression."""

# Authors: Fabio Sigrist <fabiosigrist@gmail.com>
# (the module ruses code from scikit-learn)
# License: BSD 3 clause

import numpy as np
from ridge_exten import _solve_cholesky_kernel_sparse
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model.ridge import _solve_cholesky_kernel
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import check_pairwise_arrays
import bottleneck as bn
from scipy.sparse import csc_matrix
import scipy.spatial
from sklearn.utils import check_random_state
import warnings

"""
TODO
- Add support for all pairwise_kernel implemented in scikit-learn
- Add option for more than one alpha in '_solve_cholesky_kernel_sparse'
"""

class KernelRidge(BaseEstimator, RegressorMixin):
    """Kernel ridge regression.

    Kernel ridge regression (KRR) combines ridge regression (linear least
    squares with l2-norm regularization) with the kernel trick. It thus
    learns a linear function in the space induced by the respective kernel and
    the data. For non-linear kernels, this corresponds to a non-linear
    function in the original space.

    The form of the model learned by KRR is identical to support vector
    regression (SVR). However, different loss functions are used: KRR uses
    squared error loss while support vector regression uses epsilon-insensitive
    loss, both combined with l2 regularization. In contrast to SVR, fitting a
    KRR model can be done in closed-form and is typically faster for
    medium-sized datasets. On the other  hand, the learned model is non-sparse
    and thus slower than SVR, which learns a sparse model for epsilon > 0, at
    prediction-time.

    This estimator has built-in support for multi-variate regression
    (i.e., when y is a 2d-array of shape [n_samples, n_targets]).

    Parameters
    ----------
    alpha : {float, array-like}, shape = [n_targets]
        Small positive values of alpha improve the conditioning of the problem
        and reduce the variance of the estimates.  Alpha corresponds to
        ``(2*C)^-1`` in other linear models such as LogisticRegression or
        LinearSVC. If an array is passed, penalties are assumed to be specific
        to the targets. Hence they must correspond in number.

    kernel : string or callable, default="rbf"
        Kernel mapping used internally. A callable should accept two arguments
        and the keyword arguments passed to this object as kernel_params, and
        should return a floating point number.
        
    theta : float, default=None
        Range parameter for the GW (=generalized Wendland), rbf (=Gaussian) 
        and laplace (=exponential) kernels. 
        Interpretation of the default value is left to the kernel.
    
    kernel_mat : {array-like, can be sparse}, shape = [n_samples, n_samples], 
                default=None
        Precomputed kernel matrix. This is also called gram matrix or 
        covariance matrix (for Gaussian processes)
    
    solve_kernel : {array-like}, shape = [n_samples, n_samples], default=None
        Either (i) a precomputed inverse kernel matrix or a (ii) solver that 
        calculates the weight vectors in the kernel space. The first option (i) 
        is used for dense kernel matrices and the second 
        option (ii) for sparse kernel matrices

    nystroem: boolean, default=None
        Indicates whether Nystroem sampling is used or not.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    n_components : int, detault = 100
        Number of data points used in Nystroem sampling.

    component_indices : {array-like}, shape = [n_components], default=None
        The indices of the data points used for Nystroem sampling.

    Attributes
    ----------
    dual_coef_ : array, shape = [n_samples] or [n_samples, n_targets]
        Representation of weight vector(s) in kernel space

    X_fit_ : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Training data, which is also required for prediction

    References
    ----------
    * Kevin P. Murphy
      "Machine Learning: A Probabilistic Perspective", The MIT Press
      chapter 14.4.3, pp. 492-493

    See also
    --------
    sklearn.linear_model.Ridge:
        Linear ridge regression.
    sklearn.svm.SVR:
        Support Vector Regression implemented using libsvm.

    Examples
    --------
    >>> from sklearn.kernel_ridge import KernelRidge
    >>> import numpy as np
    >>> n_samples, n_features = 10, 5
    >>> rng = np.random.RandomState(0)
    >>> y = rng.randn(n_samples)
    >>> X = rng.randn(n_samples, n_features)
    >>> clf = KernelRidge(alpha=1.0)
    >>> clf.fit(X, y) # doctest: +NORMALIZE_WHITESPACE
    KernelRidge(alpha=1.0, coef0=1, degree=3, gamma=None, kernel='linear',
                kernel_params=None)
    """
    def __init__(self, alpha=1., kernel="rbf", kernel_mat=None, 
                 solve_kernel=None, n_neighbors=50, prctg_neighbors=None, 
                 theta=None, range_adjust = 1., sparse=False, nystroem=False,
                 random_state=None, n_components = 100, 
                 component_indices=None):
        self.alpha = alpha
        self.kernel = kernel
        self.kernel_mat = kernel_mat
        self.solve_kernel = solve_kernel##Either solver for sparse=True or inverse kernel matrix for sparse=False
        self.n_neighbors = n_neighbors
        self.prctg_neighbors = prctg_neighbors
        self.theta = theta
        self.sparse = sparse
        self.range_adjust = range_adjust
        self.nystroem = nystroem
        self.random_state = random_state
        self.n_components = n_components
        self.component_indices = component_indices

    def _get_kernel(self, X, Y=None, nystroem_kernel=False):
        X, Y = check_pairwise_arrays(X, Y)
        if nystroem_kernel:##Cannot use self.nytroem since kernel needs also be computable for full data for prediction when Nystroem sampling is used
            if self.component_indices is None:
                rnd = check_random_state(self.random_state)
                n_samples = X.shape[0]
                 # get basis vectors
                if self.n_components > n_samples:
                    # XXX should we just bail?
                    n_components = n_samples
                    warnings.warn("n_components > n_samples. This is not possible.\n"
                                  "n_components was set to n_samples, which results"
                                  " in inefficient evaluation of the full kernel.")
                else:
                    n_components = self.n_components
                n_components = min(n_samples, n_components)
                self.component_indices = rnd.permutation(n_samples)[:n_components]
            X = X[self.component_indices].copy()
            d = euclidean_distances(X,X)
        else:
            d = euclidean_distances(X,Y)
        ##Get n_neighbors largest element to find range if not given
        if (self.theta is None):
            if (self.n_neighbors == "inf") | (self.n_neighbors == np.inf):##special case: chose theta such that it equals the average distance to the farest neighbor
                self.n_neighbors = X.shape[0]-1
                self.range_adjust = 1.
            if (not self.prctg_neighbors is None) & (self.n_neighbors is None): 
                self.n_neighbors = int(X.shape[0]*self.prctg_neighbors)
            if not self.n_neighbors is None:
                if self.kernel=="GW":##Choose theta such that on average every point has n_neighbors non-zero entries
                    ds=d.flatten()
                    ds=ds[~(ds==0)]##Remove diagonal
                    self.theta=bn.partition(ds, d.shape[0]*self.n_neighbors-1)[d.shape[0]*self.n_neighbors-1]
                else:##Choose theta as average distance to n_neighbors'th nearest neighbor
                    kdt = scipy.spatial.cKDTree(X)
                    dists, neighs = kdt.query(X, self.n_neighbors+1)##get distance to n_neighbors+1 nearest neighbors (incl. point itself)
                    self.theta=np.mean(dists[:, self.n_neighbors])##calculate average distance to n_neighbors'th nearest neighbor (only true neighbors excl. point itself)
            if self.kernel=="rbf": self.theta = self.theta/(self.range_adjust**0.5)##range_adjust=3 (4.6) correlation should drop to 5% (1%) at distance = theta
            if self.kernel=="laplace": self.theta = self.theta/self.range_adjust
            print("Chosen theta: " +str(round(self.theta,4)))
        if self.kernel=="GW":
            d *= -1./self.theta
            d2=d.copy()
            d+=1.
            d[d<0]=0
            d*=d
            d2*=-2
            d2+=1
            d*=d2
            ##Above code does the same as below:
    #        tmp=1-d/self.theta
    #        tmp[tmp<0]=0
    #        d=tmp**2*(1+2*d/self.theta)
        if self.kernel=="rbf":
            ##np.exp(-(d/self.theta)**2)
            d*=(1./self.theta)
            d*=-d
            np.exp(d,d)
        if self.kernel=="laplace":
            ##np.exp(-d/self.theta)
            d*=(-1./self.theta)
            np.exp(d,d)
        if self.sparse:
#            print("Sparsity ratio: " +str(round(float(100*np.sum(d>0))/X.shape[0]/X.shape[0],2))+"%")
            return csc_matrix(d)
        else:
            return d

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def fit(self, X, y=None, sample_weight=None):
        """Fit Kernel Ridge regression model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training data

        y : array-like, shape = [n_samples] or [n_samples, n_targets]
            Target values

        sample_weight : float or array-like of shape [n_samples]
            Individual weights for each sample, ignored if None is passed.

        Returns
        -------
        self : returns an instance of self.
        """
        # Convert data
        X, y = check_X_y(X, y, accept_sparse=("csr", "csc"), multi_output=True,
                         y_numeric=True)
        if sample_weight is not None and not isinstance(sample_weight, float):
            sample_weight = check_array(sample_weight, ensure_2d=False)
        if self.kernel_mat is None: 
            self.kernel_mat = self._get_kernel(X, nystroem_kernel=self.nystroem)
        alpha = np.atleast_1d(self.alpha)# XXX not needed anymore for KTBoost?

        ravel = False
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            ravel = True

        if self.nystroem: # XXX maybe this can be done better (without the need for copying)
            y = y[self.component_indices].copy()
            if not sample_weight is None: sample_weight = sample_weight[self.component_indices].copy()
            X = X[self.component_indices].copy()

        if self.sparse:
            if self.solve_kernel is None:
                K = self.kernel_mat.copy()##Need to copy since for the weighted case, the matrix gets modified
                self.dual_coef_ = _solve_cholesky_kernel_sparse(K, y, alpha,
                                                                sample_weight)
            else:
                self.dual_coef_ = self.solve_kernel(y)
        else:
            if self.solve_kernel is None:
                self.dual_coef_ = _solve_cholesky_kernel(self.kernel_mat, y, alpha,
                                                         sample_weight,copy=True)##Need to copy since for the weighted case, the matrix gets modified
            else:
                self.dual_coef_ = self.solve_kernel.dot(y)   

        if ravel:
            self.dual_coef_ = self.dual_coef_.ravel()

        self.X_fit_ = X

        return self

    def predict(self, X, training_data=False, pred_kernel_mat=None):
        """Predict using the kernel ridge model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        C : array, shape = [n_samples] or [n_samples, n_targets]
            Returns predicted values.
        """
        check_is_fitted(self, ["X_fit_", "dual_coef_"])
        if (training_data) & (not self.kernel_mat is None):
            K=self.kernel_mat
        else:
            if pred_kernel_mat is None:
                K = self._get_kernel(X, self.X_fit_)
            else:
                K = pred_kernel_mat
        if self.sparse:
            return K.dot(self.dual_coef_)
        else:
            return np.dot(K, self.dual_coef_)