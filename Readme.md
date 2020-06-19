# KTBoost - A Python Package for Boosting

This Python package implements several boosting algorithms with different combinations of base learners, optimization algorithms, and loss functions.

## Description

Concerning **base learners**, KTboost includes:

* Trees 
* Reproducing kernel Hilbert space (RKHS) ridge regression functions (i.e., posterior means of Gaussian processes)
* A combination of the two (the KTBoost algorithm) 


Concerning the **optimization** step for finding the boosting updates, the package supports:

* Gradient descent
* Newton's method (if applicable)
* A hybrid gradient-Newton version for trees as base learners (if applicable)


The package implements the following **loss functions**:

 * Continuous data ("regression"): quadratic loss (L2 loss), absolute error (L1 loss), Huber loss, quantile regression loss, Gamma regression loss, negative Gaussian log-likelihood with both the mean and the standard deviation as functions of features
* Count data ("regression"): Poisson regression loss
* (Unorderd) Categorical data ("classification"): logistic regression loss (log loss), exponential loss, cross entropy loss with softmax
* Mixed continuous-categorical data ("censored regression"): negative Tobit likelihood (the Grabit model)




## Installation

It can be **installed** using 
```
pip install -U KTBoost
```
and then loaded using 
```python
import KTBoost.KTBoost as KTBoost
```

## Author
Fabio Sigrist

## References

* Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. The Annals of Statistics, 1189-1232.
* [Sigrist, F., & Hirnschall, C. (2019).](https://arxiv.org/abs/1711.08695) Grabit: Gradient Tree Boosted Tobit Models for Default Prediction. Journal of Banking & Finance
* [Sigrist, F. (2018)](https://arxiv.org/abs/1808.03064). Gradient and Newton Boosting for Classification and Regression. arXiv preprint arXiv:1808.03064.
* [Sigrist, F. (2019).](https://arxiv.org/abs/1902.03999) KTBoost: Combined Kernel and Tree Boosting. arXiv preprint arXiv:1902.03999.


## Usage and examples
The package is build as an extension of the scikit-learn implementation of boosting algorithms and its workflow is very similar to that of scikit-learn.

The two main classes are `KTBoost.BoostingClassifier` and `KTBoost.BoostingRegressor`. The following **code examples** show how the package can be used. See also below for more information on the main parameters.

See also the [**Grabit demo**](https://github.com/fabsig/KTBoost/blob/master/examples/Grabit_demo.py) for working **examples of the Grabit model**.


#### Define models, train models, make predictions
```python
import KTBoost.KTBoost as KTBoost

################################################
## Define model (see below for more examples) ##
################################################
## Standard tree-boosting for regression with quadratic loss and hybrid gradient-Newton updates as in Friedman (2001)
model = KTBoost.BoostingRegressor(loss='ls')

##################
## Train models ##
##################
model.fit(Xtrain,ytrain)

######################
## Make predictions ##
######################
model.predict(Xpred)
```

#### More examples of models
```python
#############################
## More examples of models ##
#############################
## Boosted Tobit model, i.e. Grabit model (Sigrist and Hirnschall, 2017), 
## with lower and upper limits at 0 and 100
model = KTBoost.BoostingRegressor(loss='tobit',yl=0,yu=100)
## KTBoost algorithm (combined kernel and tree boosting) for classification with Newton updates
model = KTBoost.BoostingClassifier(loss='deviance',base_learner='combined',
                                    update_step='newton',theta=1)
## Gradient boosting for classification with trees as base learners
model = KTBoost.BoostingClassifier(loss='deviance',update_step='gradient')
## Newton boosting for classification model with trees as base learners
model = KTBoost.BoostingClassifier(loss='deviance',update_step='newton')
## Hybrid gradient-Newton boosting (Friedman, 2001) for classification with 
## trees as base learners (this is the version that scikit-learn implements)
model = KTBoost.BoostingClassifier(loss='deviance',update_step='hybrid')
## Kernel boosting for regression with quadratic loss
model = KTBoost.BoostingRegressor(loss='ls',base_learner='kernel',theta=1)
## Kernel boosting with the Nystroem method and the range parameter theta chosen 
## as the average distance to the 100-nearest neighbors (of the Nystroem samples)
model = KTBoost.BoostingRegressor(loss='ls',base_learner='kernel',nystroem=True,
                                  n_components=1000,theta=None,n_neighbors=100)
## Regression model where both the mean and the standard deviation depend 
## on the covariates / features
model = KTBoost.BoostingRegressor(loss='msr')
```

#### Feature importances and partial dependence plots
```python
#########################
## Feature importances ## (only defined for trees as base learners)
#########################
Xtrain=np.random.rand(1000,10)
ytrain=2*Xtrain[:,0]+2*Xtrain[:,1]+np.random.rand(1000)

model = KTBoost.BoostingRegressor()
model.fit(Xtrain,ytrain)
## Extract feature importances calculated as described in Friedman (2001)
feat_imp = model.feature_importances_

## Alternatively, plot feature importances directly
KTBoost.plot_feature_importances(model=model,feature_names=feature_names,maxFeat=10)

##############################
## Partial dependence plots ## (currently only implemented for trees as base learners)
##############################
from KTBoost.partial_dependence import plot_partial_dependence
import matplotlib.pyplot as plt
features = [0,1,2,3,4,5]
fig, axs = plot_partial_dependence(model,Xtrain,features,percentiles=(0,1),figsize=(8,6))
plt.subplots_adjust(top=0.9)
fig.suptitle('Partial dependence plots')

## Alternatively, get partial dependencies in numerical form
from KTBoost.partial_dependence import partial_dependence
kwargs = dict(X=Xtrain, percentiles=(0, 1))
partial_dependence(model,[0],**kwargs)
```

#### Summary of main parameters
In the following, we describe the most important parameters of the constructors of the two classes `KTBoost.BoostingClassifier` and `KTBoost.BoostingRegressor`.

* **loss** : loss function to be optimized.
    * `KTBoost.BoostingClassifier`
    {'deviance', 'exponential'}, optional (default='deviance')
    
        'deviance' refers to the logistic regression loss for binary classification, and the cross-entropy        loss with the softmax function for multiclass classification.
    
    * `KTBoost.BoostingRegressor`
    {'ls', 'lad', 'huber', 'quantile', 'poisson', 'gamma', 'tobit', 'msr'}, optional (default='ls')
    
        'ls' refers to the squarred loss. 'lad' (least absolute deviation) is a robust
        version. 'huber' is a combination of the former two. 'quantile'
        allows quantile regression (use 'alpha' to specify the quantile).
        'tobit' corresponds to the [Grabit model](https://arxiv.org/abs/1711.08695) with a Tobit loss.            'msr' is a linear regression model where both the mean and the logarithm of the standard deviation         are varying.

* **update_step** : string, default="hybrid"

    Defines how boosting updates are calculated. Use either "gradient" for gradient boosting
    or "newton" for Newton boosting (if applicable). "hybrid" uses a gradient step for finding the structur of trees and a Newton step for finding the leaf values. For kernel boosting, "hybrid" uses
    gradient descent. See the [reference paper](https://arxiv.org/abs/1808.03064) for more information.

* **base_learner** : string, default="tree"

    Base learners used in boosting updates. Choose among "tree" for trees, "kernel" for
    reproducing kernel Hilbert space (RKHS) regression functions, and "combined" for a combination of the two. See the [reference paper](https://arxiv.org/abs/1902.03999) for more information.

* **learning_rate** : float, optional (default=0.1)

    The learning rate shrinks the contribution of each base learner by 'learning_rate'.
    There is a trade-off between learning_rate and n_estimators.

* **n_estimators** : int (default=100)
    
    The number of boosting iterations to perform.

* **max_depth** : integer, optional (default=5)

    Maximum depth of the regression trees. The maximum
    depth limits the number of nodes in the tree. This value determines the interaction
    of the predictor variables.

* **min_samples_leaf** : int, float, optional (default=1)

    The minimum number of samples required to be at a leaf node:

    - If int, then consider `min_samples_leaf` as the minimum number.
    - If float, then `min_samples_leaf` is a percentage and
      `ceil(min_samples_leaf * n_samples)` are the minimum
      number of samples for each node.
      
* **min_weight_leaf** : float, optional (default=1.)

    The minimum number of weighted samples required to be at a leaf node.
    If Newton boosting is used, this corresponds to the equivalent (i.e.,
    normalized) number of weighted samples where the weights are determined
    based on the second derivatives / Hessians.

* **criterion** : string, optional (default="mse")

    The function to measure the quality of a split. Supported criteria
    are "friedman_mse" for the mean squared error with improvement
    score by Friedman, "mse" for mean squared error, and "mae" for
    the mean absolute error.
      
* **random_state** : int, RandomState instance or None, optional (default=None)

    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

* **kernel** : string, default="rbf"

    Kernel function used for kernel boosting. Currently, supports "laplace", "rbf", and "GW"
    (generalied Wendland with "smoothness parameter" mu=1).

* **theta** : float, default: 1.

    Range parameter of the kernel functions which determines how fast the kernel function
    decays with distance.

* **n_neighbors** : int, default: None

    If the range parameter 'theta' is not given, it can be determined from the data using this
    parameter. The parameter 'theta' is chosen as the average distance of the 'n_neighbors'
    nearest neighbors distances. The parameter 'range_adjust' can be used to modify this.
    If range_adjust=3 or range_adjust=4.6, 'theta' is chosen such that the kernel function has
    decayed to essentially zero (0.05 or 0.01, respectively) at the average distance of the
    'n_neighbors' nearest neighbors (for rbf and laplace kernel).

* **alphaReg** : float, default: 1.

    Regularization parameter for kernel Ridge regression boosting updates.

* **nystroem** : boolean, default=None

    Indicates whether Nystroem sampling is used or not for kernel boosting.

* **n_components** : int, detault = 100

    Number of data points used in Nystroem sampling for kernel boosting.
