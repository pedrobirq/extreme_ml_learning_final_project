import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import objects.datasets
from config import config

import utils



def train_classic_ml(model_class, params, task):
    if task == 'titanic':
        preparer = objects.datasets.TitanicDatasetPrepare(config.paths.titanic_train)
        X, y = preparer.to_xy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.general.test_size, random_state=config.general.random_state, stratify=y)

    skf = StratifiedKFold(n_splits=config.cv.k_forlds, shuffle=config.cv.shuffle, random_state=config.general.random_state)
    models = []
    train_metrics = []

    for train_index, val_index in skf.split(X_train, y_train):
        model = model_class(**params)

        X_tn, X_val = X_train[train_index], X_train[val_index]
        y_tn, y_val = y_train[train_index], y_train[val_index]

        model.fit(X_tn, y_tn)
        y_pred = model.predict(X_val)

        train_metrics.append(utils.save_cv_metrics(y_val, y_pred))
        models.append(model)

    train_metrics_avg = utils.aggregate_cv_metrics(train_metrics)

    test_metrics = []
    for model in models:
        y_pred = model.predict(X_test)
        test_metrics.append(utils.save_cv_metrics(y_test, y_pred))

    test_metrics_avg = utils.aggregate_cv_metrics(test_metrics)

    return models, train_metrics, test_metrics


MODEL_REGISTRY = {
    'LogisticRegression': LogisticRegression,
    'KNeighborsClassifier': KNeighborsClassifier,
    'DecisionTreeClassifier': DecisionTreeClassifier,
    'RandomForestClassifier': RandomForestClassifier,
    'CatBoostClassifier': CatBoostClassifier,
    'LGBMClassifier': LGBMClassifier,
    'XGBClassifier': XGBClassifier,
}


def run():
    for model_name, params in config.classic_ml_models.classification.items():
        models, train_metrics, test_metrics = train_classic_ml(MODEL_REGISTRY[model_name], params, 'titanic')

        print('\n', '=' * 10, model_name, '=' * 10)
        agg_train = utils.aggregate_cv_metrics(train_metrics)
        agg_test = utils.aggregate_cv_metrics(test_metrics)

        print('Train log\n', agg_train, '\nTest log\n', agg_test)


if __name__ == '__main__':
    run()