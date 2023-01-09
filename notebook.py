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

#Section 1
t_train , t_val = train_test_split(norm_train, test_size = 0.2, random_state = (73 + 98))

from verify_gradients import compare_gradients
#Q2
X_train = t_train.drop("contamination_level" ,axis=1, inplace=False)
y_train = t_train["contamination_level"]
# compare_gradients(X_train, y_train, deltas=np.logspace(-7, -2, 9))

#Q3
from test_lr import test_lr
X_val = t_val.drop("contamination_level" ,axis=1, inplace=False)
y_val = t_val["contamination_level"]
 
# test_lr(X_train, y_train, X_val=X_val, y_val = y_val, title="Linear Regressor Losses with different learning rates")

#Section 2 - Evaluation and Baseline
from sklearn.dummy import DummyRegressor

dummy_regr = DummyRegressor(strategy="mean")

from sklearn.model_selection import cross_validate
X_train = norm_train.drop('contamination_level', axis=1, inplace=False)
y_train = norm_train.contamination_level
results = cross_validate(dummy_regr, X_train, y=y_train, return_train_score=True, scoring = 'neg_mean_squared_error')
train_mse = results["train_score"].mean()
validation_mse = results["test_score"].mean()

print("train mse : {}, and valid mse : {}", train_mse, validation_mse)

dummy_regr.fit(X_train, y_train) #retraining dummy for later

# hyperparameter tuning of LinearRegressor
from LinearRegressor import LinearRegressor
lr_list = np.logspace(-9, -1, 9)

validation_scores = []
train_scores = []
# for lr in lr_list:
#     cur_linear_reggressor = LinearRegressor(lr)
#     results = cross_validate(cur_linear_reggressor, X_train, y=y_train, scoring='neg_mean_squared_error', return_train_score=True)
#     train_scores.append( results["train_score"].mean())
#     validation_scores.append( results["test_score"].mean())

#     print(results)
# g = plt.semilogx(lr_list, train_scores, color = 'orange', markersize = 15)
# g = plt.semilogx(lr_list, validation_scores, color = 'teal', markersize = 15)


# plt.xlabel('lr')
# plt.ylabel('score')
# plt.title('LinearRegressor Model Cross Validation Train and Validation Scores')
# plt.grid(True)
# plt.legend(["train", "validation"], loc ="upper right")
# plt.show()


from sklearn.linear_model import Lasso

validation_scores = []
train_scores = []
# alpha_list = np.logspace(-5, 5, 40)
alpha_list = np.linspace(0,1, 11)
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


best_lasso = Lasso(alpha=alpha_list[np.argmax(validation_scores)], fit_intercept=True)
best_lasso.fit(X_train, y_train)

print("---------------hello-----------------")

for i,nn in enumerate(best_lasso.coef_):
    print("feature is " + str(norm_train.columns[i]) + "value is ",best_lasso.coef_[i])
# idxs = np.argsort(best_lasso.coef_)[-5:]
# for i in  idxs:
#     print(norm_train.columns[i])

#"feature absolute value"
#coeff_array = np.sort(np.abs(best_lasso.coef_))

coefs = np.abs(best_lasso.coef_)
coefs = -np.sort(-coefs)
indexes = np.linspace(0,26,27)
ax = plt.gca()

ax.plot(indexes, coefs)

plt.axis('tight')
plt.xlabel('index')
plt.ylabel('absolute value')
plt.title("feature absolute value");
plt.show()