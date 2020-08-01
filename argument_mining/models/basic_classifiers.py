###########
# IMPORTS #
###########

# Standard Library Imports
import os
import re

# Third Party Library Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas import ExcelWriter
from pandas import ExcelFile
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
# from xgboost import XGBClassifier | ISSUES WITH DOWNLOAD ON WINDOWS!

# Local Library Imports


########
# CODE #
########

class BasicClassifiers:
    """
    "BasicClassifiers" Implements Common Model Functions From the "sklearn" Library, Such As:

        - Decision Tree
        - Logistic Regression
        - kNN
        - Naive Bayes
        - Random Forest
        - SVM
        - XGBoost

    """

    def __init__(self):
        self.train_x, self.test_x, self.train_y, self.test_y = None, None, None, None

    def load_data(self, dataset_path: str, stage_skip=0) -> pd.DataFrame():
        """
        Load and Consolidate the Dataset Into a Pandas DataFrame and Return That DataFrame

        :param dataset_path:    Path to the Dataset
        :param stage_skip:      Exclude the First "N" Stages From the Dataset
        :return:                Pandas DataFrame of the Entire Dataset
        """

        pass

    def decision_tree(self) -> None:
        """
        Decision Tree Classifier Model

        :return:    None
        """

        print('DecisionTreeClassifier')

        model = DecisionTreeClassifier()

        # Fit the Model With the Training Data
        model.fit(self.train_x, self.train_y)

        # Depth of the Decision Tree
        print('Depth of the Decision Tree :', model.get_depth())

        # Predict the Target on the Train Dataset
        predict_train = model.predict(self.train_x)
        # print('Target on train data',predict_train)

        # Accuracy Score on Train Dataset
        accuracy_train = accuracy_score(self.train_y, predict_train)
        print('accuracy_score on train dataset : ', accuracy_train)

        # Predict the Target on the Test Dataset
        predict_test = model.predict(self.test_x)
        # print('Target on test data',predict_test)

        # Accuracy Score on Test Dataset
        accuracy_test = accuracy_score(self.test_y, predict_test)
        print('accuracy_score on test dataset : ', accuracy_test)

    def knn(self) -> None:
        """
        K Nearest Neighbors Classifier Model

        :return:    None
        """

        print('KNeighborsClassifier')

        model = KNeighborsClassifier()

        # Fit the Model With the Training Data
        model.fit(self.train_x, self.train_y)

        # Number of Neighbors Used to Predict the Target
        print('The number of neighbors used to predict the target : ', model.n_neighbors)

        # Predict the Target on the Train Dataset
        predict_train = model.predict(self.train_x)
        # print('\nTarget on train data',predict_train)

        # Accuracy Score on Train Dataset
        accuracy_train = accuracy_score(self.train_y, predict_train)
        print('accuracy_score on train dataset : ', accuracy_train)

        # Predict the Target on the Test Dataset
        predict_test = model.predict(self.test_x)
        # print('Target on test data',predict_test)

        # Accuracy Score on Test Dataset
        accuracy_test = accuracy_score(self.test_y, predict_test)
        print('accuracy_score on test dataset : ', accuracy_test)

    def logistic_regression(self) -> None:
        """
        Logistic Regression Classifier Model

        :return:    None
        """

        print('Logistic Regression')

        # Fit the Model With the Training Data
        model = LogisticRegression()
        model.fit(self.train_x, self.train_y)

        # Predict the Target on the Train Dataset
        predict_train = model.predict(self.train_x)

        # Accuracy Score on Train Dataset
        accuracy_train = accuracy_score(self.train_y, predict_train)
        print('accuracy_score on train dataset : ', accuracy_train)

        predict_test = model.predict(self.test_x)

        # Accuracy Score on test dataset
        accuracy_test = accuracy_score(self.test_y, predict_test)
        print('accuracy_score on test dataset : ', accuracy_test)

    def naive_bayes(self) -> None:
        """
        Naive Bayes Classifier Model

        :return:    None
        """

        print('GaussianNB')

        model = GaussianNB()

        # Fit the Model With the Training Data
        model.fit(self.train_x, self.train_y)

        # Predict the Target on the Train Dataset
        predict_train = model.predict(self.train_x)
        # print('Target on train data',predict_train)

        # Accuracy Score on Train Dataset
        accuracy_train = accuracy_score(self.train_y, predict_train)
        print('accuracy_score on train dataset : ', accuracy_train)

        # Predict the Target on the Test Dataset
        predict_test = model.predict(self.test_x)
        # rint('Target on test data',predict_test)

        # Accuracy Score on Test Dataset
        accuracy_test = accuracy_score(self.test_y, predict_test)
        print('accuracy_score on test dataset : ', accuracy_test)

    def random_forest(self) -> None:
        """
        Random Forest Classifier Model

        :return:    None
        """

        print('RandomForestClassifier')

        model = RandomForestClassifier()

        # Fit the Model With the Training Data
        model.fit(self.train_x, self.train_y)

        # Number of Trees Used
        print('Number of Trees used : ', model.n_estimators)

        # Predict the Target on the Train Dataset
        predict_train = model.predict(self.train_x)
        # print('\nTarget on train data',predict_train)

        # Accuracy Score on Train Dataset
        accuracy_train = accuracy_score(self.train_y, predict_train)
        print('accuracy_score on train dataset : ', accuracy_train)

        # Predict the Target on the Test Dataset
        predict_test = model.predict(self.test_x)
        # print('\nTarget on test data',predict_test)

        # Accuracy Score on Test Dataset
        accuracy_test = accuracy_score(self.test_y, predict_test)
        print('accuracy_score on test dataset : ', accuracy_test)

    def svm(self) -> None:
        """
        Support Vector Machine Classifier Model

        :return:    None
        """

        print('SVM')

        model = SVC()

        # Fit the Model With the Training Data
        model.fit(self.train_x, self.train_y)

        # Predict the Target on the Train Dataset
        predict_train = model.predict(self.train_x)

        # Accuracy Score on Train Dataset
        accuracy_train = accuracy_score(self.train_y, predict_train)
        print('accuracy_score on train dataset : ', accuracy_train)

        # Predict the Target on the Test Dataset
        predict_test = model.predict(self.test_x)

        # Accuracy Score on Test Dataset
        accuracy_test = accuracy_score(self.test_y, predict_test)
        print('accuracy_score on test dataset : ', accuracy_test)

    # def xgb(self) -> None:
    #     """
    #     XGBoost Classifier Model
    #
    #     :return:
    #     """
    #
    #     print('XGB')
    #
    #     model = XGBClassifier()
    #
    #     # Fit the Model With the Training Data
    #     model.fit(self.train_x, self.train_y)
    #
    #     # Predict the Target on the Train Dataset
    #     predict_train = model.predict(self.train_x)
    #     # print('\nTarget on train data',predict_train)
    #
    #     # Accuracy Score on Train Dataset
    #     accuracy_train = accuracy_score(self.train_y, predict_train)
    #     print('accuracy_score on train dataset : ', accuracy_train)
    #
    #     # Predict the Target on the Test Dataset
    #     predict_test = model.predict(self.test_x)
    #     # print('\nTarget on test data',predict_test)
    #
    #     # Accuracy Score on Test Dataset
    #     accuracy_test = accuracy_score(self.test_y, predict_test)
    #     print('accuracy_score on test dataset : ', accuracy_test)
