import os
from data_preprocessing import preprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from lightgbm import LGBMClassifier

path_to_data = os.path.join(os.getcwd(), "biomarker_dataset.xlsx")
data = pd.read_excel(path_to_data)

X, y = preprocessing(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Hyperparameter tuning
model = LGBMClassifier(
    objective="multiclass",
    num_class=len(np.unique(y)),
    boosting_type="gbdt",
    verbose=-1,
)
kfold = KFold(n_splits=10, shuffle=True, random_state=0)
param_grid = {
    "num_leaves": [5, 20, 31],
    "learning_rate": [0.05, 0.1, 0.2],
    "n_estimators": [50, 100, 150]
}
grid_search = GridSearchCV(model, param_grid, cv=kfold, scoring="f1_weighted")
grid_search.fit(X_train, y_train)
print("Score on the train set: {:.2f}".format(grid_search.score(X_train, y_train)))
print("Score on the test set: {:.2f}".format(grid_search.score(X_test, y_test)))
print("The best parameter values: {}".format(grid_search.best_params_))