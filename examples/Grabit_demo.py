# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 18:50:49 2019

author: Fabio Sigrist
"""

import sklearn.datasets as datasets
import numpy as np
import KTBoost.KTBoost as KTBoost
import random

"""
Example 1
"""
# simulate data
random.seed(10)
n = 1000
X, lp = datasets.make_friedman3(n_samples=n)
X_test, y_test = datasets.make_friedman3(n_samples=n)
lp = lp*5+0.2
y_test = y_test*5+0.2
y=np.random.normal(loc=lp,scale=1)
# apply censoring
yu=8
yl=5
y[y>=yu]=yu
y[y<=yl]=yl

# train model and make predictions
model=KTBoost.BoostingRegressor(loss='tobit', yl=yl, yu=yu).fit(X, y)
y_pred=model.predict(X_test)
# mean square error (approx. 0.45 for n=1000)
print("Test error Grabit: " + str(((y_pred-y_test)**2).mean()))
# compare to standard least squares gradient boosting (approx. 1.1 for n=1000)
model_ls=KTBoost.BoostingRegressor(loss='ls').fit(X, y)
y_pred_ls=model_ls.predict(X_test)
print("Test error standard least square gradient boosting: " + str(((y_pred_ls-y_test)**2).mean()))

# measure time
import time
start = time.time()
model=KTBoost.BoostingRegressor(loss='tobit', yl=yl, yu=yu)
model.fit(X, y)
end = time.time()
print(end - start)
# approx. 1 sec (14.5 secs) when n = 1000 (50000) on a standard laptop


"""
Example 2: 2-d non-linear function
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def nonlin_fct(x1,x2):
    r=x1**2+x2**2
    r=np.pi*2*1*(r**0.75)
    f=2*np.cos(r)
    return(f)
def plot_2d_fct(x1,x2,y,title="2d function",elev=45,azim=120,zlim=None,filename=None):
    fig = plt.figure(figsize=(8, 7))
    ax = Axes3D(fig)
    if zlim is not None:
        ax.set_zlim3d(zlim)
        surf = ax.plot_surface(x1, x2, y, rstride=1, cstride=1,
                       cmap=plt.cm.BuPu, edgecolor='k',vmax=zlim[1])
    else:
        surf = ax.plot_surface(x1, x2, y, rstride=1, cstride=1,
                   cmap=plt.cm.BuPu, edgecolor='k')
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel('')
    #  pretty init view
    ax.view_init(elev=elev, azim=azim)
    plt.colorbar(surf)
    plt.suptitle(title)
    plt.subplots_adjust(top=0.9)
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename,dpi=200)
    
##True function
nx=100
x=np.arange(-1+1/nx,1,2/nx)
x1, x2 = np.meshgrid(x, x)
yt=nonlin_fct(x1,x2)
zlim=(-1.75,1.75)
plot_2d_fct(x1,x2,yt,title="True F",zlim=zlim)
        
# simulate data
n=10000
np.random.seed(10)  
X=np.random.rand(n,2)
X=(X-0.5)*2
y=nonlin_fct(X[:,0],X[:,1])+np.random.normal(scale=1, size=n)
# apply xensoring
yc=y.copy()
yl=np.percentile(y,q=33.33)
yu=np.percentile(y,q=66.66)
yc[y>=yu]=yu
yc[y<=yl]=yl

# train Grabit model and make predictions
model = KTBoost.BoostingRegressor(loss='tobit', yl=yl, yu=yu,sigma=1,
                                  learning_rate=0.1,n_estimators=100,max_depth=3)
model.fit(X, yc)
X_pred = np.transpose(np.array([x1.flatten(),x2.flatten()]))
y_pred = model.predict(X_pred)
plot_2d_fct(x1,x2,y_pred.reshape((100,-1)),title="Grabit",zlim=zlim)

# compare to standard least squares gradient boosting
model = KTBoost.BoostingRegressor(loss='ls',learning_rate=0.1,n_estimators=100,max_depth=3)
model.fit(X, yc)
X_pred = np.transpose(np.array([x1.flatten(),x2.flatten()]))
y_pred = model.predict(X_pred)
plot_2d_fct(x1,x2,y_pred.reshape((100,-1)),title="L2 Boosting",zlim=zlim)
