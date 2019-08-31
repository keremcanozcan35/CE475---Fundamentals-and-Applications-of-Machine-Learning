# -*- coding: utf-8 -*-
"""
Created on Sat May 11 01:33:13 2019

@author: KCO
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC,SVR
import statsmodels.formula.api as sm

data = pd.read_csv('data.csv')
X = data.values[:,:-1]
y = data.values[:,[6]]
X_train = X[:100]
y_train = y[:100,[0]]
X_test = X[100:]

reg = LinearRegression()
reg.fit(X_train,y_train,sample_weight=10)
y_pred1 = reg.predict(X_test)

X_train_temp = np.append(arr = np.ones((100,1)).astype(int), 
                         values = X_train,axis = 1)
X_temp = X_train_temp[:,[0,1,2,3,4,5,6]]
reg_OLS = sm.OLS(y_train,X_temp).fit()
reg_OLS.summary()
X_train_final = X[:100,[0,1,2,3,5]]
X_test_final = X[100:,[0,1,2,3,5]]

reg.fit(X_train_final,y_train)
y_pred2 = reg.predict(X_test_final)

reg_polynomial = PolynomialFeatures(degree = 6)
X_polynomial_train = reg_polynomial.fit_transform(X_train_final)
reg.fit(X_polynomial_train,y_train)
X_polynomial_test = reg_polynomial.fit_transform(X_test_final)
y_pred3 = reg.predict(X_polynomial_test)

reg_DT = DecisionTreeRegressor()
reg_DT.fit(X_train,y_train)
y_pred4 = reg_DT.predict(X_test)

reg_RF = RandomForestRegressor(n_estimators = 1000)
reg_RF.fit(X_train,y_train)
y_pred5 = reg_RF.predict(X_test)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.fit_transform(X_test)
SVM_classifier = SVC(kernel = 'linear')
SVM_classifier.fit(X_train_scaled,y_train)
y_pred6 = SVM_classifier.predict(X_test_scaled)

SVR_reg = SVR(kernel = 'rbf')
SVR_reg.fit(X_train_scaled, sc.fit_transform(y_train))
y_pred7 = SVR_reg.predict(X_test_scaled)
y_pred7 = sc.inverse_transform(y_pred7)

x1 = X[:,0]
x2 = X[:,1]
x3 = X[:,2]
x4 = X[:,3]
x5 = X[:,4]
x6 = X[:,5]
plt.subplot(231)
plt.scatter(x1,y,s=50,marker = 'o')
plt.title("x1-y")

plt.subplot(232)
plt.scatter(x2,y,s=50,marker = 'o')
plt.title("x2-y")

plt.subplot(233)
plt.scatter(x3,y,s=50,marker = 'o')
plt.title("x3-y")

plt.subplot(234)
plt.scatter(x4,y,s=50,marker = 'o')
plt.title("x4-y")

plt.subplot(235)
plt.scatter(x5,y,s=50,marker = 'o')
plt.title("x5-y")

plt.subplot(236)
plt.scatter(x6,y,s=50,marker = 'o')
plt.title("x6-y")

#Cross-Validation
dataset_CV = pd.read_csv("data.csv")
X_CV = dataset_CV.values[:100,:-1]
Y_CV = dataset_CV.values[:100,-1]
X_CV = X_CV[:,[0,1,2,3,5]]

from sklearn.model_selection import train_test_split,cross_val_score
X_train_CV, X_test_CV, Y_train_CV, Y_test_CV = train_test_split(X_CV,Y_CV,test_size = 0.2)

from sklearn.metrics import mean_squared_error,confusion_matrix,r2_score
from math import sqrt
regressorML = LinearRegression()
regressorML.fit(X_train_CV,Y_train_CV)
y_predML_CV = regressorML.predict(X_test_CV)
results1 = cross_val_score(estimator = regressorML,X = X_train_CV,y = Y_train_CV,cv = 10)
RmseML = sqrt(mean_squared_error(Y_test_CV, y_predML_CV))
meanML = results1.mean()

regressorPR = PolynomialFeatures(degree = 6)
X_train_polynomial_CV = regressorPR.fit_transform(X_train_CV)
X_test_polynomial_CV = regressorPR.fit_transform(X_test_CV)
LinRegPR = LinearRegression()
LinRegPR.fit(X_train_polynomial_CV,Y_train_CV)
y_predPol_CV = LinRegPR.predict(X_test_polynomial_CV)
results2 = cross_val_score(estimator = LinRegPR,X = X_train_polynomial_CV,y = Y_train_CV,cv = 10)
RmsePR = sqrt(mean_squared_error(Y_test_CV, y_predPol_CV))
meanPOL = results2.mean()

regressorDT = DecisionTreeRegressor()
regressorDT.fit(X_train_CV,Y_train_CV)
y_predDT_CV = regressorDT.predict(X_test_CV)
results3 = cross_val_score(estimator = regressorDT,X = X_train_CV,y = Y_train_CV,cv = 10)
RmseDT = sqrt(mean_squared_error(Y_test_CV, y_predDT_CV))
meanDT = results3.mean()

regressorRF = RandomForestRegressor(n_estimators = 1000)
regressorRF.fit(X_train_CV,Y_train_CV)
y_predRF_CV = regressorRF.predict(X_test_CV)
results4 = cross_val_score(estimator = regressorRF,X = X_train_CV,y = Y_train_CV,cv = 10)
RmseRF = sqrt(mean_squared_error(Y_test_CV, y_predRF_CV))
meanRF = results4.mean()
R2RF = r2_score(Y_test_CV,y_predRF_CV)

sc_CV = StandardScaler()
X_train_CV_scaled = sc_CV.fit_transform(X_train_CV)
X_test_CV_scaled = sc_CV.fit_transform(X_test_CV)
classifierSVM = SVC(kernel = 'linear')
classifierSVM.fit(X_train_CV_scaled,Y_train_CV)
y_predSVM_CV = classifierSVM.predict(X_test_CV_scaled)
results5 = cross_val_score(estimator = classifierSVM,X = X_train_CV_scaled,y = Y_train_CV,cv = 3)
RmseSVM = sqrt(mean_squared_error(Y_test_CV, y_predSVM_CV))
confusionMatrixSVM = confusion_matrix(Y_test_CV,y_predSVM_CV)
meanSVM = results5.mean()

SVR_regressor = SVR(kernel = 'rbf')
SVR_regressor.fit(X_train_CV_scaled, Y_train_CV)
results6 = cross_val_score(estimator = SVR_regressor,X = X_train_CV_scaled,y = Y_train_CV,cv = 10)
RmseSVR = sqrt(mean_squared_error(Y_test_CV, y_predSVM_CV))
meanSVR = results6.mean()
















