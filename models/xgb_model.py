import os
from data_preprocessing import preprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

path_to_data = os.path.join(os.getcwd(), "biomarker_dataset.xlsx")
data = pd.read_excel(path_to_data)

X, y = preprocessing(data)
le = LabelEncoder()
y = le.fit_transform(y)

# stratified k-fold cross validation
model = XGBClassifier()
kfold = KFold(n_splits=10, shuffle=True, random_state=0)

scores = cross_val_score(model, X, y, cv=kfold, scoring="f1_weighted")
print("Cross validation score: \n{}".format(scores))
print("Mean value: {:.3f}".format(np.mean(scores)))

# The "Depth" feature removing
X_wo_depth = X[:,1:]
scores = cross_val_score(model, X_wo_depth, y, cv=kfold, scoring="f1_weighted")
print("Cross validation score: \n{}".format(scores))
print("Mean value: {:.3f}".format(np.mean(scores)))