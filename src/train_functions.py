import pandas as pd
import numpy as np
from scipy.stats import mode
from pathlib import Path

from configs.config_titanic import config
from src.datasets.titanic import TitanicDatasetPrepare, TitanicDatasetPrepareNN
from src.models.registry import CLASSIC_ML_MODEL_REGISTRY, DL_REGISTRY, NN_ATTRIBUTES
from src.models.dnn_models import EarlyStopping
from src import utils

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, auc, roc_auc_score

import optuna

import torch
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm.auto import tqdm


# def make_folds(config):
#     raw_df = pd.read_csv(config.paths.train)
#     skf = StratifiedKFold(n_splits=config.cv.k_folds, shuffle=config.cv.shuffle, random_state=config.general.random_state)


def train_classification(model_class: str, 
                         model_params: dict,
                         preprocessing_params: dict,
                         nn_attributes: dict = None,
                         optuna_optimizing=None):

    raw_df = pd.read_csv(config.paths.train)

    skf = StratifiedKFold(n_splits=config.cv.k_folds, shuffle=config.cv.shuffle, random_state=config.general.random_state)
    cv_metrics = {'val_f1': [], 'val_roc_auc': []}

    model_name = model_class
    is_nn = False
    if model_name in CLASSIC_ML_MODEL_REGISTRY:
        model_builder = CLASSIC_ML_MODEL_REGISTRY[model_name]
    elif model_name in DL_REGISTRY:
        model_builder = DL_REGISTRY[model_name]
        is_nn = True
    else:
        raise ValueError(f"Model '{model_name}' wasn't in the registry")

    models = []
    fold_statistics = []

    # Main training loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(raw_df, raw_df[config.general.target])):

        # raw data slices
        train_fold_df = raw_df.iloc[train_idx].copy()
        val_fold_df = raw_df.iloc[val_idx].copy()

        # train data
        train_preparer = TitanicDatasetPrepare(train_fold_df)
        train_prep_df = train_preparer.prepare_dataset(**preprocessing_params)
        X_train, y_train = train_preparer.to_xy(train_prep_df)

        # val data
        val_preparer = TitanicDatasetPrepare(val_fold_df, is_train=False, statistics=train_preparer.statistics)
        val_prep_df = val_preparer.prepare_dataset(**preprocessing_params)
        X_val, y_val = val_preparer.to_xy(val_prep_df)

        if not is_nn:
            model = model_builder(**model_params)

            # model training
            model.fit(X_train, y_train)

            # model validating
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)
            
            cv_metrics['val_f1'].append(f1_score(y_val, y_pred))
            cv_metrics['val_roc_auc'].append(roc_auc_score(y_val, y_proba[:, 1]))

            models.append(model)
        else:
            model = model_builder(**model_params).to(config.general.device)

            train_nn_preparer = TitanicDatasetPrepareNN(train_prep_df)
            val_nn_preparer = TitanicDatasetPrepareNN(val_prep_df)

            train_loader = DataLoader(train_nn_preparer, batch_size=config.general.batch_size, shuffle=True)
            val_loader = DataLoader(val_nn_preparer, batch_size=config.general.batch_size, shuffle=False)

            checkpoint_filename = f"fold_{fold}_{model_name}.pt"

            early_stopping = EarlyStopping(
                checkpoint_path=Path("checkpoints") / checkpoint_filename,
                save_to_disk=True,  # Во время Optuna пишем ТОЛЬКО в RAM
                **config.early_stopping,
            )

            print('FOLD ', fold + 1)
            model, best_metrics, history = train_nn_classification(model, 
                                                     train_loader=train_loader, 
                                                     val_loader=val_loader,
                                                     early_stopping=early_stopping,
                                                     **nn_attributes
                                                     )

            for key, value in best_metrics.items():
                if key not in cv_metrics.keys():
                    cv_metrics[key] = []

                cv_metrics[key].append(value)

            models.append(model.cpu())
        
        # statistics saving
        fold_statistics.append(train_preparer.statistics)

    if optuna_optimizing is not None and optuna_optimizing in cv_metrics.keys():
        return float(np.mean(cv_metrics[optuna_optimizing]))
    elif optuna_optimizing is not None:
        raise ValueError('Not an appropriate metric')

    return cv_metrics, models, fold_statistics


def objective(trial: optuna.Trial):
    params = {
        "loss_function": "Logloss",
        "eval_metric": "F1",
        "random_seed": config.general.random_state,
        "verbose": False,
        # Число деревьев и темп обучения
        "iterations": trial.suggest_int("iterations", 300, 1200, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        # Структура деревьев
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        # Сэмплирование данных
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli"]
        ),
    }

    # Специфичные параметры для разных типов бутстрапа
    if params["bootstrap_type"] == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float(
            "bagging_temperature", 0.0, 1.0
        )
    elif params["bootstrap_type"] == "Bernoulli":
        params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

    

    f1_scores = train_classification('CatBoostClassifier', params, config.preprocessing_params, config, optuna_optimizing='f1_score')

    return f1_scores


def train_nn_classification(model,
                            optimizer_class, 
                            loss_class: str,
                            scheduler_class: str,
                            train_loader: DataLoader,
                            val_loader: DataLoader,
                            early_stopping: EarlyStopping = None,
                            ):

    optimizer = NN_ATTRIBUTES[optimizer_class](model.parameters(), lr=config.general.learning_rate)
    criterion = NN_ATTRIBUTES[loss_class]()
    
    scheduler = None
    if scheduler_class:
        if scheduler_class == 'StepLR':
            scheduler = NN_ATTRIBUTES[scheduler_class](optimizer, **config.schedulers.step_lr)
        elif scheduler_class == 'CosineAnnealingLR':
            scheduler = NN_ATTRIBUTES[scheduler_class](optimizer, **config.schedulers.cosine)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_f1': [],
        'val_f1': [],
        'train_roc_auc': [],
        'val_roc_auc': []
    }

    pbar_train = tqdm(total=len(train_loader), desc="Train", position=0, leave=True)
    pbar_val = tqdm(total=len(val_loader), desc="Val  ", position=1, leave=True)

    for epoch in range(config.general.epochs):
        pbar_train.reset(total=len(train_loader))
        pbar_val.reset(total=len(val_loader))

        model.train()
        running_train_loss = 0.0
        train_targets = []
        train_predictions = [] 
        train_probabilities = []

        for x, targets in train_loader:
            x = x.to(config.general.device)
            targets = targets.to(config.general.device)

            pred = model(x)
            loss_val = criterion(pred, targets)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            running_train_loss += loss_val.item() * len(targets)

            probs = torch.sigmoid(pred).detach()
            preds = (probs > 0.5).int()

            train_probabilities.extend(probs.flatten().tolist())
            train_predictions.extend(preds.flatten().tolist())
            train_targets.extend(targets.flatten().tolist())

            pbar_train.update(1)
            pbar_train.set_description(f"Epoch {epoch+1}/{config.general.epochs} [Train]")
            pbar_train.set_postfix({"loss": f"{loss_val.item():.4f}"})

        mean_train_loss = running_train_loss / len(train_targets)
        epoch_train_f1 = f1_score(train_targets, train_predictions)
        epoch_train_roc = roc_auc_score(train_targets, train_probabilities)

        history['train_loss'].append(mean_train_loss)
        history['train_f1'].append(epoch_train_f1)
        history['train_roc_auc'].append(epoch_train_roc)

        model.eval()
        running_val_loss = 0.0
        val_targets = []
        val_predictions = []
        val_probabilities = []

        with torch.no_grad():
            for x, targets in val_loader:
                x = x.to(config.general.device)
                targets = targets.to(config.general.device)

                pred = model(x)
                loss_val = criterion(pred, targets)

                running_val_loss += loss_val.item() * len(targets)

                probs = torch.sigmoid(pred)
                preds = (probs > 0.5).int()

                val_probabilities.extend(probs.flatten().tolist())
                val_predictions.extend(preds.flatten().tolist())
                val_targets.extend(targets.flatten().tolist())

                pbar_val.update(1)
                pbar_val.set_description(f"Epoch {epoch+1}/{config.general.epochs} [Val]")
                pbar_val.set_postfix({"loss": f"{loss_val.item():.4f}"})

        mean_val_loss = running_val_loss / len(val_targets)
        epoch_val_f1 = f1_score(val_targets, val_predictions)
        epoch_val_roc = roc_auc_score(val_targets, val_probabilities)

        history['val_loss'].append(mean_val_loss)
        history['val_f1'].append(epoch_val_f1)
        history['val_roc_auc'].append(epoch_val_roc)

        if scheduler is not None:
            scheduler.step()

        current_epoch_metrics = {
            'val_loss': mean_val_loss,
            'val_f1': epoch_val_f1,
            'val_roc_auc': epoch_val_roc,
            'train_loss': mean_train_loss,
            'train_f1': epoch_train_f1,
            'train_roc_auc': epoch_train_roc
        }

        if early_stopping is not None:
            should_stop = early_stopping(
                tracked_parameter=mean_val_loss,
                model=model,
                epoch=epoch + 1,
                metrics=current_epoch_metrics,
                optimizer=optimizer
            )
            if should_stop:
                print(f"\n[EarlyStopping] Обучение остановлено на эпохе {epoch+1}. Лучшая эпоха: {early_stopping.best_epoch}")
                break

    pbar_train.close()
    pbar_val.close()

    # best model state
    if early_stopping is not None:
        early_stopping.restore_best_weights(model)
        best_metrics = early_stopping.best_metrics
    else:
        # Если early stopping не использовался
        best_metrics = {
            'best_epoch': config.general.epochs,
            'val_loss': history['val_loss'][-1],
            'val_f1': history['val_f1'][-1],
            'val_roc_auc': history['val_roc_auc'][-1],
            'train_loss': history['train_loss'][-1],
            'train_f1': history['train_f1'][-1],
            'train_roc_auc': history['train_roc_auc'][-1]
        }

    return model, best_metrics, history


def run_all_models(models_type, preprocessing_params, config):
    if models_type == 'classic_ml_models':
        for model_class, model_params in config.classic_ml_models.items():
            cv_metrics = train_classification(model_class, model_params, preprocessing_params, config)[0]
            print('~' * 10, model_class, '~' * 10)
            print(utils.calc_cv_statistics(cv_metrics))
            print()


def make_test_predictions(models: list, fold_statistics: list, preprocessing_params, config, submission_name, nn=False):
    test_raw_df = pd.read_csv(config.paths.test)

    fold_predictions = []

    for fold_idx, (model, stats) in enumerate(zip(models, fold_statistics)):

        test_preparer = TitanicDatasetPrepare(
            test_raw_df, 
            is_train=False, 
            statistics=stats
        )
        
        test_prep_df = test_preparer.prepare_dataset(**preprocessing_params)

        if not nn:
            X_test_fold = test_preparer.to_xy(test_prep_df)
            
            y_test_pred_fold = model.predict(X_test_fold)
        else:
            test_nn_preparer = TitanicDatasetPrepareNN(test_prep_df)
            test_loader = DataLoader(test_nn_preparer)

            y_test_pred_fold = []

            with torch.no_grad():
                model.eval()
                for x in test_loader:
                    x = x.to(config.general.device)

                    pred = model(x)

                    probs = torch.sigmoid(pred)
                    preds = (probs > 0.5).int()

                    y_test_pred_fold.extend(preds.flatten().tolist())

        fold_predictions.append(y_test_pred_fold)

    final_preds = mode(fold_predictions).mode

    submission = pd.DataFrame({
        'PassengerId': test_preparer.PassengerId,
        'Survived': final_preds
    })
    submission.to_csv(config.paths.submissions + submission_name + '.csv', index=False)