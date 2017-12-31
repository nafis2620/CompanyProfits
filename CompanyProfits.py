# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 08:17:13 2017

@author: nafis
So I used a data-preprocessing template from my Machine Learning Class and the way I approached this method is using simple linear regression to compare what could be expected from a certain price point and what is the actual outcome.
The reason I used simple linear regression is because I only have indpendent and one dependent variable.
"""

# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('CompanyProfits.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values




# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

# Predicting the Test set results
# Visualising the Training set results
plt.scatter(X, y, color = 'orange')
plt.plot(X, regressor.predict(X), color = 'green')
plt.title('Customer Profits')
plt.xlabel('Amount')
plt.ylabel('Price')
plt.show()

