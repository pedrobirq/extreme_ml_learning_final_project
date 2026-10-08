# from src import utils
# from models.dnn_models import SimpNN, MoreLayersNN, ImprovedNN
import pandas as pd
import numpy as np
from scipy.stats import mode

from src.datasets.titanic import TitanicDatasetPrepare, TitanicDatasetPrepareNN
from src.models.registry import CLASSIC_ML_MODEL_REGISTRY, DL_REGISTRY, NN_ATTRIBUTES
from src.models.dnn_models import EarlyStopping
from src import utils

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, auc, roc_auc_score

# from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge, ElasticNet
# from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, VotingRegressor, StackingRegressor

# from catboost import CatBoostClassifier, CatBoostRegressor
# from lightgbm import LGBMClassifier, LGBMRegressor
# from xgboost import XGBClassifier, XGBRegressor

import torch
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm.auto import tqdm



# def train_classic_ml(model_class, params, task, X_train, X_test, y_train):
#     """
#     Train function for classic ml models
#     Input:
#         model_class - classname of a model to train;
#         params - a dictionary of model parameters;
#         task - titanic/houses;
#         data: X_train, X_test, y_train
#     Output:
#         train_metrics - metrics depended on a given task;
#         y_test - model prediction on a test dataset.
#     """

#     # CV initialization
#     if task == 'titanic':
#         kf = StratifiedKFold(n_splits=config.cv.k_folds, shuffle=config.cv.shuffle, random_state=config.general.random_state)
#     elif task == 'houses':
#         kf = KFold(n_splits=config.cv.k_folds, shuffle=config.cv.shuffle, random_state=config.general.random_state)

#     models = []
#     train_metrics = []

#     # Main training loop
#     for train_index, val_index in kf.split(X_train, y_train):
#         model = model_class(**params)

#         X_tn, X_val = X_train[train_index], X_train[val_index]
#         y_tn, y_val = y_train[train_index], y_train[val_index]

#         model.fit(X_tn, y_tn)

#         y_pred = model.predict(X_val)

#         if task == 'titanic':
#             train_metrics.append(utils.save_cv_metrics(y_val, y_pred))
#         elif task == 'houses':
#             train_metrics.append(utils.save_cv_regression_metrics(y_val, y_pred))
#         models.append(model)

#     if task == 'titanic':
#         y_test = utils.make_classification_prediction(X_test, models)
#     elif task == 'houses': 
#         y_test = utils.make_regression_prediction(X_test, models)

#     return train_metrics, y_test


def train_classification(model_class: str, 
                         model_params: dict,
                         preprocessing_params: dict,
                         config,
                         nn_attributes: dict = None):

    raw_df = pd.read_csv(config.paths.train)

    skf = StratifiedKFold(n_splits=config.cv.k_folds, shuffle=config.cv.shuffle, random_state=config.general.random_state)
    cv_metrics = dict()

    dl = False
    if model_class in CLASSIC_ML_MODEL_REGISTRY:
        model = CLASSIC_ML_MODEL_REGISTRY[model_class](**model_params)
    elif model_class in DL_REGISTRY:
        model = DL_REGISTRY[model_class](**model_params).to(config.general.device)
        df = True

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

        if not df:
            cv_metrics['f1_score'] = []
            cv_metrics['roc_auc_score'] = []

            # model training
            model.fit(X_train, y_train)

            # model validating
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)
            
            cv_metrics['f1_score'].append(f1_score(y_val, y_pred))
            cv_metrics['roc_auc_score'].append(roc_auc_score(y_val, y_proba[:, 1]))
        else:
            train_nn_preparer = TitanicDatasetPrepareNN(train_prep_df)
            val_nn_preparer = TitanicDatasetPrepareNN(val_prep_df)

            train_loader = DataLoader(train_nn_preparer, batch_size=config.general.batch_size, shuffle=True)
            val_loader = DataLoader(val_nn_preparer, batch_size=config.general.batch_size, shuffle=False)

            early_stopping = EarlyStopping(checkpoint_name=f'{fold}_fold_{model_class}.pt')

            print('FOLD ', fold)
            model, best_metrics, history = train_nn_classification(model, 
                                                     train_loader=train_loader, 
                                                     val_loader=val_loader,
                                                     config=config,
                                                     early_stopping=early_stopping,
                                                     **nn_attributes
                                                     )
            print(fold)

            for key, value in best_metrics.items():
                if key not in cv_metrics.keys():
                    cv_metrics[key] = []

                cv_metrics[key].append(value)
        
        # model and statistics saving
        models.append(model)
        fold_statistics.append(train_preparer.statistics)

    return cv_metrics, models, fold_statistics


def train_nn_classification(model,
                            optimizer_class, 
                            loss_class: str,
                            scheduler_class: str,
                            train_loader: DataLoader,
                            val_loader: DataLoader,
                            config,
                            early_stopping: EarlyStopping = None,
                            ):

    optimizer = NN_ATTRIBUTES[optimizer_class](model.parameters(), lr=config.general.learning_rate)
    criterion = NN_ATTRIBUTES[loss_class]()
    
    scheduler = None
    if scheduler_class:
        if scheduler_class == 'StepLR':
            scheduler = NN_ATTRIBUTES[scheduler_class](optimizer, **config.schedulers.step_lr)

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


# def run_classification(data: ):
#     pass

# def train_NN(model_class, params, task, train_loader, test_loader, val_loader, optimizer_class, loss_class, epochs, verbose=False, scheduler_class=None, early_stopping=None):
#     """
#     Train function for dNN models.
#     Input:
#         Model_class - classname of a model to train;
#         params - model parameters;
#         task - titanik/houses;
#         dataloaders (train, test, val);
#         optimizer_class - classname of a model optimizer;
#         loss_class - classname of a loss function;
#         epochs - number of train epochs;
#         verbose - True/False;
#         scheduler_class - classname of a model scheduler;
#         early_stopping - EarlyStopping class object.
#     Output:
#         log - metrics depended on a given task;
#         y_test_predicted - model predictions on a test dataset
#     """
#     train_loss = []
#     val_loss = []
#     lr_list = []

#     if task == 'titanic':
#         train_f1 = []
#         train_accuracy = []
#         train_precision = []
#         train_recal = []

#         val_f1 = []
#         val_accuracy = []
#         val_precision = []
#         val_recal = []
#     elif task == 'houses':
#         train_mse = []
#         train_mae = []
#         train_r2 = []
#         train_rmse = []

#         val_mse = []
#         val_mae = []
#         val_r2 = []
#         val_rmse = []

#     # Model and its attributes preparation
#     model = model_class(**params).to(config.general.device)
#     optimizer = optimizer_class(model.parameters(), lr=config.general.learning_rate)
#     loss = loss_class()
#     if scheduler_class:
#         scheduler = scheduler_class(optimizer, **config.schedulers.step_lr)

#     pbar_train = tqdm(total=len(train_loader), desc="Train", position=0, leave=True)
#     pbar_val = tqdm(total=len(val_loader), desc="Val  ", position=1, leave=True)

#     for epoch in range(epochs):

#         pbar_train.reset(total=len(train_loader))
#         pbar_val.reset(total=len(val_loader))

#         model.train()
#         running_train_loss = []       
#         train_targets = []
#         train_predictions = [] 
#         for x, targets in train_loader:
#             x = x.to(config.general.device)

#             targets = targets.to(config.general.device)

#             # Forward pass
#             pred = model(x)

#             curr_loss = loss(pred, targets)

#             # Backward pass
#             optimizer.zero_grad()
#             curr_loss.backward()

#             optimizer.step()

#             running_train_loss.append(curr_loss.item())

#             if task == 'titanic':
#                 prob = torch.sigmoid(pred)
#                 pred = (prob > 0.5).int()

#             train_predictions.extend(pred.squeeze().tolist())
#             train_targets.extend(targets.squeeze().tolist())

#             pbar_train.update(1)
#             if verbose:
#                 pbar_train.set_description(f"Epoch {epoch+1}/{epochs} [Train]")
#                 pbar_train.set_postfix({"loss": f"{curr_loss.item():.4f}"})

#         # Training metrics calculation
#         if task == 'titanic':
#             train_metrics =  utils.save_cv_metrics(train_targets, train_predictions)
#         elif task == 'houses':
#             train_metrics = utils.save_cv_regression_metrics(train_targets, train_predictions)

#         mean_train_loss = sum(running_train_loss) / len(running_train_loss)
#         train_loss.append(mean_train_loss)

#         if task == 'titanic':
#             train_accuracy.append(train_metrics['accuracy'])
#             train_precision.append(train_metrics['precision'])
#             train_recal.append(train_metrics['recal'])
#             train_f1.append(train_metrics['f1'])
#         elif task == 'houses':
#             train_mse.append(train_metrics['mse'])
#             train_mae.append(train_metrics['mae'])
#             train_r2.append(train_metrics['r2'])
#             train_rmse.append(train_metrics['rmse'])

#         # Validation
#         model.eval()
#         running_val_loss = []
#         val_targets = []
#         val_predictions = []
#         with torch.no_grad():
#             for x, targets in val_loader:
#                 x = x.to(config.general.device)

#                 targets = targets.to(config.general.device)

#                 pred = model(x)

#                 curr_loss = loss(pred, targets)

#                 running_val_loss.append(curr_loss.item())

#                 if task == 'titanic':
#                     prob = torch.sigmoid(pred)
#                     pred = (prob > 0.5).int()

#                 val_predictions.extend(pred.squeeze().tolist())
#                 val_targets.extend(targets.squeeze().tolist())

#                 pbar_val.update(1)
#                 if verbose:
#                     pbar_val.set_description(f"Epoch {epoch+1}/{epochs} [Val]")
#                     pbar_val.set_postfix({"loss": f"{curr_loss.item():.4f}"})

#         # Validation metrics calculation
        
#         if task == 'titanic':
#             val_metrics =  utils.save_cv_metrics(val_targets, val_predictions)
#         elif task == 'houses':
#             val_metrics = utils.save_cv_regression_metrics(val_targets, val_predictions)

#         mean_val_loss = sum(running_val_loss) / len(running_val_loss)
#         val_loss.append(mean_val_loss)

#         if task == 'titanic':
#             val_accuracy.append(val_metrics['accuracy'])
#             val_precision.append(val_metrics['precision'])
#             val_recal.append(val_metrics['recal'])
#             val_f1.append(val_metrics['f1'])
#         elif task == 'houses':
#             val_mse.append(val_metrics['mse'])
#             val_mae.append(val_metrics['mae'])
#             val_r2.append(val_metrics['r2'])
#             val_rmse.append(val_metrics['rmse'])

#         if early_stopping and early_stopping(mean_val_loss):
#             pbar_train.close()
#             pbar_val.close()
#             print(f"Обучение остановлено на {epoch+1} эпохе.")
#             break

#         if scheduler_class is not None:
#             scheduler.step()    
#             curr_lr = scheduler._last_lr[0]   
#             lr_list.append(curr_lr)

#     pbar_train.close()
#     pbar_val.close()

#     if task == 'titanic':
#         log = {'train loss': train_loss, 'train accuracy': train_accuracy, 'train precision': train_precision, 'train recal': train_recal, 'train f1': train_f1,
#            'val loss': val_loss, 'val accuracy': val_accuracy, 'val precision': val_precision, 'val recal': val_recal, 'val f1': val_f1, 'learning rates': lr_list}
#     elif task == 'houses':
#         log = {'train loss': train_loss, 'train mse': train_mse, 'train mae': train_mae, 'train r2': train_r2, 'train rmse': train_rmse, 
#                'val loss': val_loss, 'val mse': val_mse, 'val mae': val_mae, 'val r2': val_r2, 'val rmse': val_rmse, 'learning rates': lr_list}

#     # Predictions on a test dataset 
#     model.eval()
#     y_test_predicted = []
#     with torch.no_grad():
#         for x in test_loader:
#             x = x.to(config.general.device)
#             logits = model(x) 
#             if task == 'houses':
#                 y_pred = torch.expm1(logits)
#             elif task == 'titanic':
#                 y_prob = torch.sigmoid(logits)
#                 y_pred = (y_prob > 0.5).int()
#             y_test_predicted.extend(y_pred.squeeze().tolist())
            
#     return log, y_test_predicted


# # Dictionary that converts model's classname to it's actual class
# MODEL_REGISTRY = {
#     'LogisticRegression': LogisticRegression,
#     'KNeighborsClassifier': KNeighborsClassifier,
#     'DecisionTreeClassifier': DecisionTreeClassifier,
#     'RandomForestClassifier': RandomForestClassifier,
#     'VotingClassifier': VotingClassifier,
#     'StackingClassifier': StackingClassifier,
#     'CatBoostClassifier': CatBoostClassifier,
#     'LGBMClassifier': LGBMClassifier,
#     'XGBClassifier': XGBClassifier,
#     'SimpNN': SimpNN,
#     'MoreLayersNN': MoreLayersNN,
#     'ImprovedNN': ImprovedNN,
#     'LinearRegression': LinearRegression,
#     'Lasso': Lasso,
#     'Ridge': Ridge,
#     'ElasticNet': ElasticNet,
#     'KNeighborsRegressor': KNeighborsRegressor,
#     'CatBoostRegressor': CatBoostRegressor,
#     'LGBMRegressor': LGBMRegressor,
#     'XGBRegressor': XGBRegressor,
#     'VotingRegressor': VotingRegressor,
#     'StackingRegressor': StackingRegressor
# }


# def run(task, train_preparer, test_preparer, save_predictions):
#     """
#     Main training fuction. Takes data preparer classes for train dataset and test dataset; task that is needed to perform and the argument of saving models' predictions or not.
#     """
#     df_train = train_preparer.prepare_dataset()
#     df_test = test_preparer.prepare_dataset(train_preparer.statistics)

#     X_t_train, y_t_train = train_preparer.to_xy()
#     X_t_test = test_preparer.to_xy()

#     if task == 'titanic':
#         test_indexes = test_preparer.PassengerId
#     elif task == 'houses':
#         test_indexes = test_preparer.Id

#     data = {'X_train': X_t_train, 'X_test': X_t_test, 'y_train': y_t_train}
#     train_loader, val_loader, test_loader = utils.make_dataloaders(task, df_train, df_test)

#     if task == 'titanic':

#         for model_name, params in config.classic_ml_models.classification.items():
#             train_metrics, y_test = train_classic_ml(MODEL_REGISTRY[model_name], params, task, **data)

#             print('\n', '=' * 10, model_name, '=' * 10)
#             agg_train = utils.aggregate_cv_metrics(train_metrics)

#             print('Train log\n', agg_train)
#             if save_predictions:
#                 utils.save_predictions(y_test, test_indexes, f'submissions/{task}/{model_name}', column_names=['PassengerId', config.targets.titanic])

#         for ensemble_name, params in config.ensembles.classification.items():
#             estimators = [(model_name, MODEL_REGISTRY[model_name](**params)) for model_name, params in config.classic_ml_models.classification.items()]

#             ensemble_params = dict(params.copy())
#             ensemble_params['estimators'] = estimators
#             if ensemble_name == 'StackingClassifier':
#                 ensemble_params['final_estimator'] = MODEL_REGISTRY[config.ensembles.stacking_classification_final_estimator]()
#             train_metrics, y_test = train_classic_ml(MODEL_REGISTRY[ensemble_name], ensemble_params, task, **data)

#             print('\n', '=' * 10, ensemble_name, '=' * 10)
#             agg_train = utils.aggregate_cv_metrics(train_metrics)

#             print('Train log\n', agg_train)
#             if save_predictions:
#                 utils.save_predictions(y_test, test_indexes, f'submissions/{task}/{ensemble_name}', column_names=['PassengerId', config.targets.titanic])

#         for model_name, params in config.nn_models.classification.items():
#             print('\n', '=' * 10, model_name, '=' * 10)
#             early_stopping = utils.EarlyStopping(patience=config.early_stopping.patience, threshold=config.early_stopping.threshold)
#             log, y_test_predicted = train_NN(MODEL_REGISTRY[model_name], params, task, train_loader, test_loader, val_loader, 
#                         Adam, BCEWithLogitsLoss, config.general.epochs, verbose=True, scheduler_class=StepLR, early_stopping=early_stopping)
#             agg_metrics = utils.metrics_to_string('validation', log['val loss'][-1], log['val accuracy'][-1], log['val precision'][-1], log['val recal'][-1], log['val f1'][-1])

#             print('Train log\n', agg_metrics)
#             if save_predictions:
#                 utils.save_predictions(y_test_predicted, test_indexes, f'submissions/{task}/{model_name}', column_names=['PassengerId', config.targets.titanic])

#     elif task == 'houses':

#         for model_name, params in config.classic_ml_models.regression.items():
#             train_metrics, y_test = train_classic_ml(MODEL_REGISTRY[model_name], params, task, **data)

#             print('\n', '=' * 10, model_name, '=' * 10)
#             agg_train = utils.aggregate_cv_metrics(train_metrics)

#             print('Train log\n', agg_train)
#             if save_predictions:
#                 utils.save_predictions(y_test, test_indexes, f'submissions/{task}/{model_name}', column_names=['Id', config.targets.houses])

#         for ensemble_name, params in config.ensembles.regression.items():
#             estimators = [(model_name, MODEL_REGISTRY[model_name](**params)) for model_name, params in config.classic_ml_models.regression.items()]

#             ensemble_params = dict(params.copy())
#             ensemble_params['estimators'] = estimators
#             if ensemble_name == 'StackingRegressor':
#                 ensemble_params['final_estimator'] = MODEL_REGISTRY[config.ensembles.stacking_regression_final_estimator]()
#             train_metrics, y_test = train_classic_ml(MODEL_REGISTRY[ensemble_name], ensemble_params, task, **data)

#             print('\n', '=' * 10, ensemble_name, '=' * 10)
#             agg_train = utils.aggregate_cv_metrics(train_metrics)

#             print('Train log\n', agg_train)
#             if save_predictions:
#                 utils.save_predictions(y_test, test_indexes, f'submissions/{task}/{ensemble_name}', column_names=['Id', config.targets.houses])

#         for model_name, params in config.nn_models.regression.items():
#             print('\n', '=' * 10, model_name, '=' * 10)
#             early_stopping = utils.EarlyStopping(patience=config.early_stopping.patience, threshold=config.early_stopping.threshold)
#             log, y_test_predicted = train_NN(MODEL_REGISTRY[model_name], params, task, train_loader, test_loader, val_loader, 
#                         Adam, L1Loss, config.general.epochs, verbose=True, scheduler_class=StepLR, early_stopping=early_stopping)
#             agg_metrics = utils.regression_metrics_to_string('validation', log['val loss'][-1], log['val mse'][-1], log['val mae'][-1], log['val r2'][-1], log['val rmse'][-1])

#             print('Train log\n', agg_metrics)
#             if save_predictions:
#                 utils.save_predictions(y_test_predicted, test_indexes, f'submissions/{task}/{model_name}', column_names=['Id', config.targets.houses])