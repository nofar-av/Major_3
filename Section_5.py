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

from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import cross_validate
print("hello")

validation_scores = []
train_scores = []
# random_state = (73 + 98) , n_components=100
gamma_list = np.logspace(-2, 1, 6)
for gamma in gamma_list:
    forest_regressr = RandomForestRegressor(max_depth=2, random_state=(73 + 98))

    rbf_feature = RBFSampler(gamma=gamma, random_state=(73 + 98))

    X_features = rbf_feature.fit_transform(X_train)
    # forest_regressr.fit(X_features, y_train)
    results = cross_validate(forest_regressr, X_features, y=y_train, scoring='neg_mean_squared_error', return_train_score=True)
    train_scores.append( results["train_score"].mean())
    validation_scores.append( results["test_score"].mean())
    print(gamma)

    
g = plt.semilogx(gamma_list, train_scores, color = 'orange', markersize = 15)
g = plt.semilogx(gamma_list, validation_scores, color = 'teal', markersize = 15)


plt.xlabel('gamma')
plt.ylabel('score')
plt.title('RandomForestRegressor Model Cross Validation Train and Validation Scores')
plt.grid(True)
plt.legend(["train", "validation"], loc ="upper right")
plt.show()

