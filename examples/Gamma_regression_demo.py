# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 11:54:35 2021

@author: Fabio Sigrist

Small demo of Gamma regression
"""

import sklearn.datasets as datasets
import numpy as np
import KTBoost.KTBoost as KTBoost
import matplotlib.pyplot as plt

# Simulate data
np.random.seed(1)
n = 10000
X, lp = datasets.make_friedman3(n_samples=n)
X_test, lp_test = datasets.make_friedman3(n_samples=n)
lp = np.exp(lp/10)
lp_test = np.exp(lp_test/10)
y = np.random.gamma(scale=lp, shape=1)
plt.hist(y)
plt.show()

# Train model and make predictions
model = KTBoost.BoostingRegressor(loss='gamma').fit(X, y)
pred_gamma = model.predict(X_test)
# mean square error (approx. 0.026)
print("Test error of Gamma boosting: " + str(((pred_gamma-lp_test)**2).mean()))
plt.scatter(pred_gamma, lp_test)
plt.title("True vs. predicted values")
plt.show()

# Compare to predictions of classical least squares gradient boosting
model = KTBoost.BoostingRegressor(loss='ls').fit(X, y)
pred_ls = model.predict(X_test)
# mean square error (approx. 0.039)
print("Test error of standard least squares gradient boosting: " + str(((pred_ls-lp_test)**2).mean()))
plt.scatter(pred_gamma, pred_ls)
plt.title("Least squares gradient boosting vs. Gamma boosting predicted values")
plt.show()

# Compare to Tweedie boosting
model = KTBoost.BoostingRegressor(loss='tweedie').fit(X, y)
pred_tw = model.predict(X_test)
# mean square error (approx. 0.022)
print("Test error of Tweedie boosting: " + str(((pred_tw-lp_test)**2).mean()))
