
# -------------------------------------------------------------------------------------------
#                                                 libraries
# -------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
import os

# -------------------------------------------------------------------------------------------
#                                                 paths
# -------------------------------------------------------------------------------------------

path_train = 'Data/train.csv'
path_test = 'Data/test.csv'

# -------------------------------------------------------------------------------------------
#                                                 functions
# -------------------------------------------------------------------------------------------

def load_data(path):
    # we are going to drop the index, since is not relevant for our analysis
    df = pd.read_csv(path).drop(columns = 'Unnamed: 0')
    
    # calculate the average of student's score
    df['average'] = (df['math score'] + df['reading score'] + df['writing score'] )/ 3
    
    return df
def get_dummies(X):
    """
    returns the X features with dummies
    """
    return pd.get_dummies(X).drop(columns=['gender_male', 'lunch_free/reduced','test preparation course_none'])

def get_x_y(path):
    """Splits the dataset in X and Y"""

    df = load_data(path)
    
    # Split in X and Y
    X = df.drop(columns='parental level of education')
    y = df['parental level of education']

    X_dummies = get_dummies(X)

    # oversample train data
    oversample = SMOTE()
    X_train_os, y_train_os = oversample.fit_resample(X_dummies, y)
    
    return X_train_os, y_train_os


def make_predictions(X_train, y_train, X_test):
    
    scaler = StandardScaler()
    X_train_os_sc = scaler.fit_transform(X_train, y_train)
    X_test_sc = scaler.transform(X_test)
    
    clf = MLPClassifier(random_state=1, max_iter=2000).fit(X_train_os_sc, y_train)
    print(f'f1_train: {f1_score(y_train, clf.predict(X_train_os_sc), average="macro")} ')
    
    y_pred = clf.predict(X_test_sc)
    
    return y_pred

def convert_to_dict(y_pred, path):
    """
    converts the predictions into a jason file
    """
    index = list(pd.read_csv(path)['Unnamed: 0'].values.astype('str'))
    list_predictions = list(y_pred)


    target = dict(zip(index, list_predictions))

    # here I had to convert the predictions into an Int, since json does not recognize int32 or int64
    for key, values in target.items():
        target[key] = int(values)

    dictionary = {'target': target}

    return dictionary


# -------------------------------------------------------------------------------------------
#                                                  MAIN
# -------------------------------------------------------------------------------------------

X_train, y_train = get_x_y(path_train)
X_test = load_data(path_test)
X_test_dummies = get_dummies(X_test)
predictions = make_predictions(X_train, y_train, X_test_dummies)

dict = convert_to_dict(predictions, path_test)

print(dict)

with open('predictions.json', 'w') as fp:
    json.dump(dict, fp)

    
    