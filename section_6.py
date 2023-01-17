import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#a - Data loading
df = pd.read_csv(r'HW3_data.csv')

#b - partition data
from sklearn.model_selection import train_test_split
train , test = train_test_split(df, test_size = 0.2, random_state = (73 + 98))

from prepare import prepare_data
norm_train = prepare_data(train, train)
norm_test = prepare_data(train, test)

X_train = norm_train.drop("contamination_level" ,axis=1, inplace=False)
y_train = norm_train["contamination_level"]

X_test = norm_test.drop("contamination_level" ,axis=1, inplace=False)
y_test = norm_test["contamination_level"]

from LinearRegressor import LinearRegressor

linear_regressor = LinearRegressor(0.0063)
linear_regressor.fit(X_train, y_train)

print("linear regressor test score :", linear_regressor.score(X_test, y_test))

from sklearn.linear_model import Lasso

lasso_regressor = Lasso(alpha=0.3, fit_intercept=True)
lasso_regressor.fit(X_train, y_train)

print("lasso regressor test score :", lasso_regressor.score(X_test, y_test,scoring = 'neg_mean_squared_error'))

from sklearn.dummy import DummyRegressor

dummy_regressor = DummyRegressor(strategy="mean")
dummy_regressor.fit(X_train, y_train)
dummy_regressor.score(X_test, y_test)
print("dummy regressor test score :", dummy_regressor.score(X_test, y_test, scoring = 'neg_mean_squared_error'))

from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler,StandardScaler
important_features = ['PCR_05'] 
very_important_feature = ['PCR_01']
still_important_feature = ['sugar_levels']

forest_regressr = RandomForestRegressor(min_samples_leaf=1, n_estimators=200, random_state=(73 + 98))

rbf_feature = RBFSampler(gamma=0.01, random_state=(73 + 98))
ct = ColumnTransformer(transformers=[('rbf', rbf_feature, important_features), 
    ('rbf2', rbf_feature, very_important_feature),
    ('rbf3', rbf_feature, still_important_feature)], remainder = "passthrough")

clf=Pipeline(steps=[("preprocessor", ct),("classifier", forest_regressr)])
clf.fit(X_train, y_train)

print("linear regressor test score :", clf.score(X_test, y_test, scoring = 'neg_mean_squared_error'))
