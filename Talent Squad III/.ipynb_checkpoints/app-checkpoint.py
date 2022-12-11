
# Libraries
import pandas as pd
import os
from sklearn.metrics import confusion_matrix,f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

def load_dataset(filename):
    """
    Return a dataset from a name given as parameter
    :param filename: name of file
    :return: pandas dataframe
    """
    path = os.path.join(os.curdir,'Data',filename)
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print('File does not exists')
        return None


def train_data(filename):
    df = load_dataset('train.csv')

    # craft new feature as average score
    df['average_score'] = (df['math score'] + df['reading score'] + df['writing score'])/ 3

    # split between X and Y
    X = df.drop(columns=['Unnamed: 0','parental level of education'])
    y = df['parental level of education']






train_data('train.csv')