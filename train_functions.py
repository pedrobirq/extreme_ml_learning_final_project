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

import torch
from tqdm import tqdm



def train_classic_ml(model_class, params, task, X_train, X_test, y_train):
    # if task == 'titanic':
    #     preparer = objects.datasets.TitanicDatasetPrepare(config.paths.titanic_train)
    #     X, y = preparer.to_xy()

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.general.test_size, random_state=config.general.random_state, stratify=y)

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

    # train_metrics_avg = utils.aggregate_cv_metrics(train_metrics)

    if task == 'titanic':
        y_test = utils.make_classification_prediction(X_test, models)

    # test_metrics = []
    # for model in models:
    #     y_pred = model.predict(X_test)
    #     test_metrics.append(utils.save_cv_metrics(y_test, y_pred))

    # test_metrics_avg = utils.aggregate_cv_metrics(test_metrics)

    return models, train_metrics, y_test


MODEL_REGISTRY = {
    'LogisticRegression': LogisticRegression,
    'KNeighborsClassifier': KNeighborsClassifier,
    'DecisionTreeClassifier': DecisionTreeClassifier,
    'RandomForestClassifier': RandomForestClassifier,
    'CatBoostClassifier': CatBoostClassifier,
    'LGBMClassifier': LGBMClassifier,
    'XGBClassifier': XGBClassifier,
}


def train_NN(model_class, params, task, train_loader, test_loader, val_loader, optimizer, loss, epochs, verbose=False, scheduler=None):
    train_loss = []
    train_f1 = []
    train_accuracy = []
    train_precision = []
    train_recal = []

    val_loss = []
    val_f1 = []
    val_accuracy = []
    val_precision = []
    val_recal = []

    lr_list = []

    model = model_class(**params)

    for epoch in range(epochs):
        model.train()
        train_loop = tqdm(train_loader, leave=True)
        running_train_loss = []        # значения лоса на обучении
        train_targets = []
        train_predictions = [] 
        for x, targets in train_loop:
            # Подготовка данных
            x = x.to(config.general.device)

            targets = targets.to(config.general.device)

            # Прямой проход и расчет лоса
            pred = model(x)
            curr_loss = loss(pred, targets)

            # Обратный проход
            optimizer.zero_grad()
            loss.backward()

            # Шаг оптимизации
            optimizer.step()

            # Сохранение значения лоса
            running_train_loss.append(curr_loss.item())

            train_predictions.extend(pred.reshape((1, -1)).tolist())
            train_targets.extend(targets.reshape((1, -1)).tolist())

        # Вычисление метрики f1
        # TODO: implement f1 score
        running_train_f1 = 0

        mean_train_loss = sum(running_train_loss) / len(running_train_loss)

        # Вывод средней ошибки на прогресбар tqdm
        if verbose:
            train_loop.set_description(f"Epoch [{epoch+1}/{epochs}], train_loss={mean_train_loss:.4f}, train f1={running_train_f1:.4f}")

        train_loss.append(mean_train_loss)
        train_f1.append(running_train_f1)

        # Оценка тренировки модели
        with torch.no_grad():
            running_val_loss = []
            val_targets = []
            val_predictions = []
            val_loop = tqdm(val_loader)
            for x, targets in val_loop:
                x = x.reshape(-1, 28 * 28).to(config.general.device)

                targets = targets.reshape(-1).to(torch.int32)
                targets = torch.eye(10)[targets].to(config.general.device)

                pred = model(x)
                curr_loss = loss(pred, targets)

                running_val_loss.append(loss.item())

                val_predictions.extend(pred.reshape((1, -1)).tolist())
                val_targets.extend(targets.reshape((1, -1)).tolist())

        # Вычисление метрик
        val_metrics = utils.save_cv_metrics(val_targets, val_predictions)
        running_val_f1 = 0

        mean_val_loss = sum(running_val_loss) / len(running_val_loss)

        # Вывод средней ошибки на прогресбар tqdm
        if verbose:
            val_loop.set_description(utils.metrics_to_string(epoch, epochs, 'validation', mean_val_loss, **val_metrics))

        val_loss.append(mean_val_loss)
        val_f1.append(val_metrics['f1'])
        val_accuracy.append(val_metrics['accuracy'])
        val_precision.append(val_metrics['precision'])
        val_recal.append(val_metrics['recal'])

    if scheduler is not None:
        scheduler.step(mean_val_loss)    # для ReduceLROnPlayeau
        curr_lr = scheduler._last_lr[0]   
        lr_list.append(curr_lr)


def run(task: str, data: dict, test_indexes):
    for model_name, params in config.classic_ml_models.classification.items():
        models, train_metrics, y_test = train_classic_ml(MODEL_REGISTRY[model_name], params, task, **data)

        print('\n', '=' * 10, model_name, '=' * 10)
        agg_train = utils.aggregate_cv_metrics(train_metrics)
        # agg_test = utils.aggregate_cv_metrics(test_metrics)

        print('Train log\n', agg_train)
        if task == 'titanic':
            utils.save_predictions(y_test, test_indexes, f'objects/{task}/{model_name}', column_names=['PassengerId', 'Survived'])

if __name__ == '__main__':
    run()