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
t_train , t_test = train_test_split(norm_train, test_size = 0.2, random_state = (73 + 98))

from verify_gradients import compare_gradients
#Q2
X_train = t_train.drop("contamination_level" ,axis=1, inplace=False)
y_train = t_train["contamination_level"]
#compare_gradients(X_train, y_train, deltas=np.logspace(-7, -2, 9))

#Q3
from test_lr import test_lr
 
#test_lr(X_train, y_train, X_val=X_train, y_val = y_train, title="Linear Regressor Losses with different learning rates")

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
max_iter = 2000

fig, axs = plt.subplots(3, 3, sharey=True, figsize=(20, 12))
plt.suptitle("hyperparameter tuning of LinearRegressor", fontsize=32)
plt.tight_layout()
fig.subplots_adjust(hspace=0.5, top=0.9)

axs = np.ravel(axs)
for i, lr in enumerate(lr_list):
    cur_linear_reggressor = LinearRegressor(lr)
    train_losses, val_losses = cur_linear_reggressor.fit_with_logs(X_train, y_train, keep_losses=True, X_val=X_train, y_val=y_train, max_iter = max_iter)
    print('lr size = '+str(lr)+', Best train loss = '+str(min(train_losses))+', Best validation loss = '+str(min(val_losses)))

    iterations = np.arange(max_iter + 1)
    axs[i].semilogy(iterations, train_losses, label="Train")
    axs[i].semilogy(iterations, val_losses, label="Validation")
    axs[i].grid(alpha=0.5)
    axs[i].legend()
    axs[i].set_title('lr = '+str(lr))
    axs[i].set_xlabel('iteration')
    axs[i].set_ylabel('MSE')

plt.show()
