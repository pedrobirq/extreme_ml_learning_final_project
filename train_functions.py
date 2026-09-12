import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import objects.datasets
from config import config



def train_classic_ml(model_class, params, task):
    if task == 'titanic':
        preparer = objects.datasets.TitanicDatasetPrepare(config.paths.titanic_train)
        X, y = preparer.to_xy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.general.test_size, random_state=config.general.random_state, stratify=y)

        skf = StratifiedKFold(n_splits=config.cv.k_forlds, shuffle=config.cv.shuffle, random_state=config.general.random_state)
        models = []
        metrics = []

        for train_index, val_index in skf.split(X_train, y_train):
            model = model_class(**params)

            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            metrics.append({'Accuracy': accuracy_score(y_val, y_pred), 'Precision': precision_score(y_val, y_pred), 'Recal': recall_score(y_val, y_pred), 'F1': f1_score(y_val, y_pred)})
            models.append(model)

        metrics_df = pd.DataFrame(metrics)
        metrics_avg = pd.DataFrame(metrics_df.to_numpy().mean(axis=0).reshape(1, -1), columns=metrics_df.columns)

        metrics_test = []
        for model in models:
            y_pred = model.predict(X_test)
            metrics_test.append({'Accuracy': accuracy_score(y_test, y_pred), 'Precision': precision_score(y_test, y_pred), 'Recal': recall_score(y_test, y_pred), 'F1': f1_score(y_test, y_pred)})

        metrics_test_df = pd.DataFrame(metrics_test)
        metrics_test_avg = pd.DataFrame(metrics_test_df.to_numpy().mean(axis=0).reshape(1, -1), columns=metrics_test_df.columns)

        print(f'Train: \n{metrics_avg},\n\n\nTest: \n{metrics_test_avg}')



if __name__ == '__main__':
    train_classic_ml(LogisticRegression, {'random_state': config.general.random_state}, 'titanic')