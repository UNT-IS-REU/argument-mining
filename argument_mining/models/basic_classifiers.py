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
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
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
        - kNN
        - Logistic Regression
        - MLP
        - Naive Bayes
        - Random Forest
        - SVM
        - XGBoost

    """

    def __init__(self):
        self.train_x, self.test_x, self.train_y, self.test_y = None, None, None, None
        self.train_vectors, self.test_vectors = None, None
        self.df = pd.DataFrame()
        self.target_names = ['Fact', 'Issue', 'Rule/Law/Holding', 'Analysis', 'Conclusion', 'Invalid Sentence']

    def load_data(self, dataset_path: str, stage_start=0, stage_end=float('inf')) -> None:
        """
        Load and Consolidate the Dataset Into a Pandas DataFrame
        Split The DataFrame Into Train and Test Sets

        :param dataset_path:    Path to the Dataset
        :param stage_start:     Select From Which Stage You Would Like Adding Data
        :param stage_end:       Select To Which Stage You Would Like Adding Data
        :return:                Pandas DataFrame Containing All The Best Labels
        """

        stages = sorted(os.listdir(dataset_path))  # List All Stages in Dataset Folder

        # Loop Through All Selected Stages
        for stage in stages:
            stage_path = os.path.join(dataset_path, stage)
            teams = sorted(os.listdir(stage_path))
            stage_num = stage[-1]
            if stage_start <= int(stage_num) <= stage_end:
                # Loop Through All Teams In That Stage
                for team in teams:
                    team_path = os.path.join(stage_path, team)
                    team_num = team[-1]
                    argument_labels_filename = "".join(['s', stage_num,
                                                        '_t', team_num,
                                                        '_best_labels.csv'])
                    csv_file_path = os.path.join(team_path, argument_labels_filename)
                    temp_df = pd.read_csv(csv_file_path, encoding='ISO-8859-1')
                    self.df = pd.concat([self.df, temp_df])

        # Split The DataFrame Into Train and Test Sets
        train, test = train_test_split(self.df, test_size=0.2, random_state=99,
                                       shuffle=True)  # 80% training and 20% test
        self.train_x, self.train_y = train.loc[:, 'text'].values, train.loc[:, 'label_id'].values
        self.test_x, self.test_y = test.loc[:, 'text'].values, test.loc[:, 'label_id'].values

        # Vectorize Data
        vectorizer = TfidfVectorizer()
        self.train_vectors = vectorizer.fit_transform(self.train_x)
        self.test_vectors = vectorizer.transform(self.test_x)

    def decision_tree(self) -> None:
        """
        Decision Tree Classifier Model

        :return:    None
        """

        print('DecisionTreeClassifier\n')
        model = DecisionTreeClassifier()

        # Fit the Model With the Training Data
        model.fit(self.train_vectors, self.train_y)

        # Predict the Target on the Train Dataset and Print Accuracy Score
        predict_train = model.predict(self.train_vectors)
        print("Model Train Accuracy:", accuracy_score(self.train_y, predict_train))

        # Predict the Target on the Test Dataset and Print Accuracy Score
        predict_test = model.predict(self.test_vectors)
        print("Model Test Accuracy:", accuracy_score(self.test_y, predict_test))

        # Diagnostics
        # print('Depth of the Decision Tree :', model.get_depth())
        # print("Classification Report:\n", classification_report(self.test_y, predicted, target_names=self.target_names), '\n')
        # print("Confusion Matrix: [Actual Value X Predicted Value]\n\n", confusion_matrix(self.test_y, predicted))

    def knn(self) -> None:
        """
        K Nearest Neighbors Classifier Model

        :return:    None
        """

        print('KNeighborsClassifier\n')
        model = KNeighborsClassifier()

        # Fit the Model With the Training Data
        model.fit(self.train_vectors, self.train_y)

        # Predict the Target on the Train Dataset and Print Accuracy Score
        predict_train = model.predict(self.train_vectors)
        print("Model Train Accuracy:", accuracy_score(self.train_y, predict_train))

        # Predict the Target on the Test Dataset and Print Accuracy Score
        predict_test = model.predict(self.test_vectors)
        print("Model Test Accuracy:", accuracy_score(self.test_y, predict_test))

        # Diagnostics
        # print('The number of neighbors used to predict the target : ', model.n_neighbors)
        # print("Classification Report:\n", classification_report(self.test_y, predicted, target_names=self.target_names), '\n')
        # print("Confusion Matrix: [Actual Value X Predicted Value]\n\n", confusion_matrix(self.test_y, predicted))

    def logistic_regression(self) -> None:
        """
        Logistic Regression Classifier Model

        :return:    None
        """

        print('Logistic Regression\n')
        model = LogisticRegression()

        # Fit the Model With the Training Data
        model.fit(self.train_vectors, self.train_y)

        # Predict the Target on the Train Dataset and Print Accuracy Score
        predict_train = model.predict(self.train_vectors)
        print("Model Train Accuracy:", accuracy_score(self.train_y, predict_train))

        # Predict the Target on the Test Dataset and Print Accuracy Score
        predict_test = model.predict(self.test_vectors)
        print("Model Test Accuracy:", accuracy_score(self.test_y, predict_test))

        # Diagnostics
        # print("Classification Report:\n", classification_report(self.test_y, predicted, target_names=self.target_names), '\n')
        # print("Confusion Matrix: [Actual Value X Predicted Value]\n\n", confusion_matrix(self.test_y, predicted))

    def mlp(self) -> None:
        """
        MLP Neural Network Classifier

        :return:    None
        """
        print("MLP\n")
        model = MLPClassifier()

        # Fit the Model With the Training Data
        model.fit(self.train_vectors, self.train_y)

        # Predict the Target on the Train Dataset and Print Accuracy Score
        predict_train = model.predict(self.train_vectors)
        print("Model Train Accuracy:", accuracy_score(self.train_y, predict_train))

        # Predict the Target on the Test Dataset and Print Accuracy Score
        predict_test = model.predict(self.test_vectors)
        print("Model Test Accuracy:", accuracy_score(self.test_y, predict_test))

        # Diagnostics
        # print("Classification Report:\n", classification_report(self.test_y, predicted, target_names=self.target_names), '\n')
        # print("Confusion Matrix: [Actual Value X Predicted Value]\n\n", confusion_matrix(self.test_y, predicted))

    def naive_bayes(self) -> None:
        """
        Naive Bayes Classifier Model

        :return:    None
        """

        print('MultinomialNB\n')
        model = MultinomialNB()

        # Fit the Model With the Training Data
        model.fit(self.train_vectors, self.train_y)

        # Predict the Target on the Train Dataset and Print Accuracy Score
        predict_train = model.predict(self.train_vectors)
        print("Model Train Accuracy:", accuracy_score(self.train_y, predict_train))

        # Predict the Target on the Test Dataset and Print Accuracy Score
        predict_test = model.predict(self.test_vectors)
        print("Model Test Accuracy:", accuracy_score(self.test_y, predict_test))

        # Diagnostics
        # print("Classification Report:\n", classification_report(self.test_y, predicted, target_names=self.target_names), '\n')
        # print("Confusion Matrix: [Actual Value X Predicted Value]\n\n", confusion_matrix(self.test_y, predicted))

    def random_forest(self) -> None:
        """
        Random Forest Classifier Model

        :return:    None
        """

        print('RandomForestClassifier\n')
        model = RandomForestClassifier()

        # Fit the Model With the Training Data
        model.fit(self.train_vectors, self.train_y)

        # Predict the Target on the Train Dataset and Print Accuracy Score
        predict_train = model.predict(self.train_vectors)
        print("Model Train Accuracy:", accuracy_score(self.train_y, predict_train))

        # Predict the Target on the Test Dataset and Print Accuracy Score
        predict_test = model.predict(self.test_vectors)
        print("Model Test Accuracy:", accuracy_score(self.test_y, predict_test))

        # Diagnostics
        # print('Number of Trees used : ', model.n_estimators)
        # print("Classification Report:\n", classification_report(self.test_y, predicted, target_names=self.target_names), '\n')
        # print("Confusion Matrix: [Actual Value X Predicted Value]\n\n", confusion_matrix(self.test_y, predicted))

    def svm(self) -> None:
        """
        Support Vector Machine Classifier Model

        :return:    None
        """

        print('SVM\n')
        model = SVC()

        # Fit the Model With the Training Data
        model.fit(self.train_vectors, self.train_y)

        # Predict the Target on the Train Dataset and Print Accuracy Score
        predict_train = model.predict(self.train_vectors)
        print("Model Train Accuracy:", accuracy_score(self.train_y, predict_train))

        # Predict the Target on the Test Dataset and Print Accuracy Score
        predict_test = model.predict(self.test_vectors)
        print("Model Test Accuracy:", accuracy_score(self.test_y, predict_test))

        # Diagnostics
        # print("Classification Report:\n", classification_report(self.test_y, predicted, target_names=self.target_names), '\n')
        # print("Confusion Matrix: [Actual Value X Predicted Value]\n\n", confusion_matrix(self.test_y, predicted))

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
