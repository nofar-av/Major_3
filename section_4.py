import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# from matplotlib import pylab
# params = {'xtick.labelsize': 18,
#  'ytick.labelsize': 18,
#  'axes.titlesize' : 22,
#  'axes.labelsize' : 20,
#  'legend.fontsize': 18,
#  'legend.title_fontsize': 22,
#  'figure.titlesize': 24
#  }
# pylab.rcParams.update(params)


#a - Data loading
df = pd.read_csv(r'HW3_data.csv')

#b - partition data
from sklearn.model_selection import train_test_split
train , test = train_test_split(df, test_size = 0.2, random_state = (73 + 98)) 

#c - data preprocessing 
from prepare import prepare_data

norm_train = prepare_data(train, train)
norm_test = prepare_data(train, test)

subset_train = norm_train[["PCR_01", "PCR_05", "contamination_level"]]

from plot3d import plot3d
# plot3d(subset_train, "PCR_01", "PCR_05", "contamination_level", title = "PCR_01, PCR_05 and contamination_level")

from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_validate

X_train = subset_train.drop("contamination_level", axis = 1, inplace = False)
y_train = subset_train.contamination_level

validation_scores = []
train_scores = []
alpha_list = np.logspace(-10, 2, 40)
#alpha_list = np.linspace(0,1, 11)
for alpha in alpha_list:
    lasso = Lasso(alpha=alpha, fit_intercept=True)
    lasso.fit(X_train, y_train)
    results = cross_validate(lasso, X_train, y=y_train, scoring='neg_mean_squared_error', return_train_score=True)
    train_scores.append( results["train_score"].mean())
    validation_scores.append( results["test_score"].mean())

g = plt.semilogx(alpha_list, train_scores, color = 'orange', markersize = 15)
g = plt.semilogx(alpha_list, validation_scores, color = 'teal', markersize = 15)


plt.xlabel('lambda')
plt.ylabel('score')
plt.title('LinearRegressor (Lasso) Model Cross Validation Train and Validation Scores')
plt.grid(True)
plt.legend(["train", "validation"], loc ="upper right")
plt.show()

print(train_scores)
print(validation_scores)
print(alpha_list[np.argmax(validation_scores)])
print(np.max(validation_scores))
print(train_scores[np.argmax(validation_scores)])


best_lasso = Lasso(alpha=0.16, fit_intercept=True)
best_lasso.fit(X_train, y_train)


#plot3d(subset_train, "PCR_01", "PCR_05", "contamination_level", predictions = best_lasso.predict(X_train), title = "Predictions of Lasso (alpha = 0.16) of contamination_level")

#Task
Lambda = 0.16#WHAT TODO
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

poly_reg = Pipeline([('feature_mapping', PolynomialFeatures(2)),
                    ('normalization', MinMaxScaler(feature_range=(-1,1))),
                    ('Lasso', Lasso(alpha=Lambda, fit_intercept=True))])

poly_reg.fit(X_train, y_train)

#plot3d(subset_train, "PCR_01", "PCR_05", "contamination_level", predictions = poly_reg.predict(X_train), title = "Predictions of Lasso (polynomial) of contamination_level")


#Q17
from sklearn.model_selection import cross_validate


validation_scores = []
train_scores = []
alpha_list = np.logspace(-3, 0, 7)
# alpha_list = np.linspace(0,1, 11)


for alpha in alpha_list:
    lasso = Pipeline([('feature_mapping', PolynomialFeatures(2)),
                    ('normalization', MinMaxScaler(feature_range=(-1,1))),
                    ('Lasso', Lasso(alpha=alpha, fit_intercept=True))])
    lasso.fit(X_train, y_train)
    results = cross_validate(lasso, X_train, y=y_train, scoring='neg_mean_squared_error', return_train_score=True)
    train_scores.append( results["train_score"].mean())
    validation_scores.append( results["test_score"].mean())
    plot3d(subset_train, "PCR_01", "PCR_05", "contamination_level", predictions = lasso.predict(X_train), title = "Predictions of Lasso (polynomial) of contamination_level "+ str(alpha))

g = plt.semilogx(alpha_list, train_scores, color = 'orange', markersize = 15)
g = plt.semilogx(alpha_list, validation_scores, color = 'teal', markersize = 15)


plt.xlabel('lambda')
plt.ylabel('score')
plt.title('LinearRegressor (Lasso) Model Cross Validation Train and Validation Scores')
plt.grid(True)
plt.legend(["train", "validation"], loc ="upper right")
plt.show()
