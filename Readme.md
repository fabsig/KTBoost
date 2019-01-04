# KTBoost

This package implements several boosting algorithm. In particular, this 
includes tree and kernel boosting as well as the combined KTBoost algorithm. The package
supports both gradient and Newton boosting updates as well as a hybrid version of the two
for trees. Further, the package implements several loss functions which inlucdes 
the Tobit likelihood (i.e. the Grabit model).
The package is an extenion of scikit-learn and re-uses code from
scikit-learn.