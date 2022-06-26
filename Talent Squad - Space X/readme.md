# Talent Squad - Data Science I

<img src="https://media0.giphy.com/media/wPyDWwurn8XEWdR9ol/giphy.gif" width = 800>

---------------------------------------------------------------------------------
"Approximately 300 sensors are attached to the rocket and mobile launcher to detect, record and transmit information." SpaceX is launching a new rocket chain with a new business purpose, rocket-commerce. In order to achieve this, Elon Musk wants to predict the state of the rockets in order to dynamically adjust costs. For this, it has provided us with the data of several sensors and their status. The objective? Create a model that is able to predict the state.

The objective of this challenge will be to help Elon by performing predictive modeling from a dataset that contains the measurements made by his sensors and types.

## Dataset Features:

**Features:** The dataset contains 6 features in 6 columns, which are the parameters measured from the different sensors. These correspond to the vibrations detected in the rocket.

**Target:** The target corresponds to the 'label' that classifies the types of states of the rocket based on the features measured by the sensors.
* Target 0 corresponds to Stable
* Target 1 corresponds to Light Turbulence
* Target 2 corresponds to Moderate Turbulence
* Target 3 corresponds to Severe Turbulence
* Target 4 corresponds to Extreme Turbulence


## Task

The objective of the challenge will be to make a predictive model that allows knowing the type of eruption that a rocket will have based on the vibrations measured by the sensors.

Once the predictive model has been created and trained, it will have to be used with the features of the test dataset 'space_X_test.csv'. These predictions will have to be delivered in csv format as in the example. Where only one column will have to appear in which the first row is any text and the predictions start in row 2.

The quality of the prediction will be measured from the f1-score (macro)

## Libraries

To perform the prediction and analysis, you should make the following imports:

```python
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
```

## Models

To determinate our optimal model, we performed trials with the following predictors:

**Random Forest Classifier**

Random forest is a supervised learning method, meaning there are labels for and mappings between our input and outputs. It can be used for classification tasks like determining the species of a flower based on measurements like petal length and color, or it can used for regression tasks like predicting tomorrow’s weather forecast based on historical weather data. A random forest—as the name suggests—consists of multiple decision trees each of which outputs a prediction

**Stochastic Gradient Decent Classifier**

Stochastic Gradient Descent (SGD) is a simple yet very efficient approach to fitting linear classifiers and regressors under convex loss functions such as (linear) Support Vector Machines and Logistic Regression. Even though SGD has been around in the machine learning community for a long time, it has received a considerable amount of attention just recently in the context of large-scale learning.

**Support Vector Classifier**

An SVM classifier, or support vector machine classifier, is a type of machine learning algorithm that can be used to analyze and classify data. A support vector machine is a supervised machine learning algorithm that can be used for both classification and regression tasks. The Support vector machine classifier works by finding the hyperplane that maximizes the margin between the two classes. The Support vector machine algorithm is also known as a max-margin classifier. 

**Gradient Boost Classifier**

GB builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. In each stage n_classes_ regression trees are fit on the negative gradient of the loss function, e.g. binary or multiclass log loss. Binary classification is a special case where only a single regression tree is induced.

**Grid Search CV + SVC**

Exhaustive search over specified parameter values for an estimator.<br>
Important members are fit, predict.<br>
GridSearchCV implements a “fit” and a “score” method. It also implements “score_samples”, “predict”, “predict_proba”, “decision_function”, “transform” and “inverse_transform” if they are implemented in the estimator used.<br>
The parameters of the estimator used to apply these methods are optimized by cross-validated grid-search over a parameter grid