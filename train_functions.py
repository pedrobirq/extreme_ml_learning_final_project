import utils
from config import config

from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, VotingRegressor, StackingRegressor
from objects.dnn_models import SimpNN, MoreLayersNN, ImprovedNN

from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

import torch
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm.auto import tqdm



def train_classic_ml(model_class, params, task, X_train, X_test, y_train):
    """
    Train function for classic ml models
    Input:
        model_class - classname of a model to train;
        params - a dictionary of model parameters;
        task - titanic/houses;
        data: X_train, X_test, y_train
    Output:
        train_metrics - metrics depended on a given task;
        y_test - model prediction on a test dataset.
    """

    # CV initialization
    if task == 'titanic':
        kf = StratifiedKFold(n_splits=config.cv.k_folds, shuffle=config.cv.shuffle, random_state=config.general.random_state)
    elif task == 'houses':
        kf = KFold(n_splits=config.cv.k_folds, shuffle=config.cv.shuffle, random_state=config.general.random_state)

    models = []
    train_metrics = []

    # Main training loop
    for train_index, val_index in kf.split(X_train, y_train):
        model = model_class(**params)

        X_tn, X_val = X_train[train_index], X_train[val_index]
        y_tn, y_val = y_train[train_index], y_train[val_index]

        model.fit(X_tn, y_tn)

        y_pred = model.predict(X_val)

        if task == 'titanic':
            train_metrics.append(utils.save_cv_metrics(y_val, y_pred))
        elif task == 'houses':
            train_metrics.append(utils.save_cv_regression_metrics(y_val, y_pred))
        models.append(model)

    if task == 'titanic':
        y_test = utils.make_classification_prediction(X_test, models)
    elif task == 'houses': 
        y_test = utils.make_regression_prediction(X_test, models)

    return train_metrics, y_test



def train_NN(model_class, params, task, train_loader, test_loader, val_loader, optimizer_class, loss_class, epochs, verbose=False, scheduler_class=None, early_stopping=None):
    """
    Train function for dNN models.
    Input:
        Model_class - classname of a model to train;
        params - model parameters;
        task - titanik/houses;
        dataloaders (train, test, val);
        optimizer_class - classname of a model optimizer;
        loss_class - classname of a loss function;
        epochs - number of train epochs;
        verbose - True/False;
        scheduler_class - classname of a model scheduler;
        early_stopping - EarlyStopping class object.
    Output:
        log - metrics depended on a given task;
        y_test_predicted - model predictions on a test dataset
    """
    train_loss = []
    val_loss = []
    lr_list = []

    if task == 'titanic':
        train_f1 = []
        train_accuracy = []
        train_precision = []
        train_recal = []

        val_f1 = []
        val_accuracy = []
        val_precision = []
        val_recal = []
    elif task == 'houses':
        train_mse = []
        train_mae = []
        train_r2 = []
        train_rmse = []

        val_mse = []
        val_mae = []
        val_r2 = []
        val_rmse = []

    # Model and its attributes preparation
    model = model_class(**params).to(config.general.device)
    optimizer = optimizer_class(model.parameters(), lr=config.general.learning_rate)
    loss = loss_class()
    if scheduler_class:
        scheduler = scheduler_class(optimizer, **config.schedulers.step_lr)

    pbar_train = tqdm(total=len(train_loader), desc="Train", position=0, leave=True)
    pbar_val = tqdm(total=len(val_loader), desc="Val  ", position=1, leave=True)

    for epoch in range(epochs):

        pbar_train.reset(total=len(train_loader))
        pbar_val.reset(total=len(val_loader))

        model.train()
        running_train_loss = []       
        train_targets = []
        train_predictions = [] 
        for x, targets in train_loader:
            x = x.to(config.general.device)

            targets = targets.to(config.general.device)

            # Forward pass
            pred = model(x)

            curr_loss = loss(pred, targets)

            # Backward pass
            optimizer.zero_grad()
            curr_loss.backward()

            optimizer.step()

            running_train_loss.append(curr_loss.item())

            if task == 'titanic':
                prob = torch.sigmoid(pred)
                pred = (prob > 0.5).int()

            train_predictions.extend(pred.squeeze().tolist())
            train_targets.extend(targets.squeeze().tolist())

            pbar_train.update(1)
            if verbose:
                pbar_train.set_description(f"Epoch {epoch+1}/{epochs} [Train]")
                pbar_train.set_postfix({"loss": f"{curr_loss.item():.4f}"})

        # Training metrics calculation
        if task == 'titanic':
            train_metrics =  utils.save_cv_metrics(train_targets, train_predictions)
        elif task == 'houses':
            train_metrics = utils.save_cv_regression_metrics(train_targets, train_predictions)

        mean_train_loss = sum(running_train_loss) / len(running_train_loss)
        train_loss.append(mean_train_loss)

        if task == 'titanic':
            train_accuracy.append(train_metrics['accuracy'])
            train_precision.append(train_metrics['precision'])
            train_recal.append(train_metrics['recal'])
            train_f1.append(train_metrics['f1'])
        elif task == 'houses':
            train_mse.append(train_metrics['mse'])
            train_mae.append(train_metrics['mae'])
            train_r2.append(train_metrics['r2'])
            train_rmse.append(train_metrics['rmse'])

        # Validation
        model.eval()
        running_val_loss = []
        val_targets = []
        val_predictions = []
        with torch.no_grad():
            for x, targets in val_loader:
                x = x.to(config.general.device)

                targets = targets.to(config.general.device)

                pred = model(x)

                curr_loss = loss(pred, targets)

                running_val_loss.append(curr_loss.item())

                if task == 'titanic':
                    prob = torch.sigmoid(pred)
                    pred = (prob > 0.5).int()

                val_predictions.extend(pred.squeeze().tolist())
                val_targets.extend(targets.squeeze().tolist())

                pbar_val.update(1)
                if verbose:
                    pbar_val.set_description(f"Epoch {epoch+1}/{epochs} [Val]")
                    pbar_val.set_postfix({"loss": f"{curr_loss.item():.4f}"})

        # Validation metrics calculation
        
        if task == 'titanic':
            val_metrics =  utils.save_cv_metrics(val_targets, val_predictions)
        elif task == 'houses':
            val_metrics = utils.save_cv_regression_metrics(val_targets, val_predictions)

        mean_val_loss = sum(running_val_loss) / len(running_val_loss)
        val_loss.append(mean_val_loss)

        if task == 'titanic':
            val_accuracy.append(val_metrics['accuracy'])
            val_precision.append(val_metrics['precision'])
            val_recal.append(val_metrics['recal'])
            val_f1.append(val_metrics['f1'])
        elif task == 'houses':
            val_mse.append(val_metrics['mse'])
            val_mae.append(val_metrics['mae'])
            val_r2.append(val_metrics['r2'])
            val_rmse.append(val_metrics['rmse'])

        if early_stopping and early_stopping(mean_val_loss):
            pbar_train.close()
            pbar_val.close()
            print(f"Обучение остановлено на {epoch+1} эпохе.")
            break

        if scheduler_class is not None:
            scheduler.step()    
            curr_lr = scheduler._last_lr[0]   
            lr_list.append(curr_lr)

    pbar_train.close()
    pbar_val.close()

    if task == 'titanic':
        log = {'train loss': train_loss, 'train accuracy': train_accuracy, 'train precision': train_precision, 'train recal': train_recal, 'train f1': train_f1,
           'val loss': val_loss, 'val accuracy': val_accuracy, 'val precision': val_precision, 'val recal': val_recal, 'val f1': val_f1, 'learning rates': lr_list}
    elif task == 'houses':
        log = {'train loss': train_loss, 'train mse': train_mse, 'train mae': train_mae, 'train r2': train_r2, 'train rmse': train_rmse, 
               'val loss': val_loss, 'val mse': val_mse, 'val mae': val_mae, 'val r2': val_r2, 'val rmse': val_rmse, 'learning rates': lr_list}

    # Predictions on a test dataset 
    model.eval()
    y_test_predicted = []
    with torch.no_grad():
        for x in test_loader:
            x = x.to(config.general.device)
            logits = model(x) 
            if task == 'houses':
                y_pred = torch.expm1(logits)
            elif task == 'titanic':
                y_prob = torch.sigmoid(logits)
                y_pred = (y_prob > 0.5).int()
            y_test_predicted.extend(y_pred.squeeze().tolist())
            
    return log, y_test_predicted


# Dictionary that converts model's classname to it's actual class
MODEL_REGISTRY = {
    'LogisticRegression': LogisticRegression,
    'KNeighborsClassifier': KNeighborsClassifier,
    'DecisionTreeClassifier': DecisionTreeClassifier,
    'RandomForestClassifier': RandomForestClassifier,
    'VotingClassifier': VotingClassifier,
    'StackingClassifier': StackingClassifier,
    'CatBoostClassifier': CatBoostClassifier,
    'LGBMClassifier': LGBMClassifier,
    'XGBClassifier': XGBClassifier,
    'SimpNN': SimpNN,
    'MoreLayersNN': MoreLayersNN,
    'ImprovedNN': ImprovedNN,
    'LinearRegression': LinearRegression,
    'Lasso': Lasso,
    'Ridge': Ridge,
    'ElasticNet': ElasticNet,
    'KNeighborsRegressor': KNeighborsRegressor,
    'CatBoostRegressor': CatBoostRegressor,
    'LGBMRegressor': LGBMRegressor,
    'XGBRegressor': XGBRegressor,
    'VotingRegressor': VotingRegressor,
    'StackingRegressor': StackingRegressor
}


def run(task, train_preparer, test_preparer, save_predictions):
    """
    Main training fuction. Takes data preparer classes for train dataset and test dataset; task that is needed to perform and the argument of saving models' predictions or not.
    """
    df_train = train_preparer.prepare_dataset()
    df_test = test_preparer.prepare_dataset(train_preparer.statistics)

    X_t_train, y_t_train = train_preparer.to_xy()
    X_t_test = test_preparer.to_xy()

    if task == 'titanic':
        test_indexes = test_preparer.PassengerId
    elif task == 'houses':
        test_indexes = test_preparer.Id

    data = {'X_train': X_t_train, 'X_test': X_t_test, 'y_train': y_t_train}
    train_loader, val_loader, test_loader = utils.make_dataloaders(task, df_train, df_test)

    if task == 'titanic':

        for model_name, params in config.classic_ml_models.classification.items():
            train_metrics, y_test = train_classic_ml(MODEL_REGISTRY[model_name], params, task, **data)

            print('\n', '=' * 10, model_name, '=' * 10)
            agg_train = utils.aggregate_cv_metrics(train_metrics)

            print('Train log\n', agg_train)
            if save_predictions:
                utils.save_predictions(y_test, test_indexes, f'objects/{task}/{model_name}', column_names=['PassengerId', config.targets.titanic])

        for ensemble_name, params in config.ensembles.classification.items():
            estimators = [(model_name, MODEL_REGISTRY[model_name](**params)) for model_name, params in config.classic_ml_models.classification.items()]

            ensemble_params = dict(params.copy())
            ensemble_params['estimators'] = estimators
            if ensemble_name == 'StackingClassifier':
                ensemble_params['final_estimator'] = MODEL_REGISTRY[config.ensembles.stacking_classification_final_estimator]()
            train_metrics, y_test = train_classic_ml(MODEL_REGISTRY[ensemble_name], ensemble_params, task, **data)

            print('\n', '=' * 10, ensemble_name, '=' * 10)
            agg_train = utils.aggregate_cv_metrics(train_metrics)

            print('Train log\n', agg_train)
            if save_predictions:
                utils.save_predictions(y_test, test_indexes, f'objects/{task}/{ensemble_name}', column_names=['PassengerId', config.targets.titanic])

        for model_name, params in config.nn_models.classification.items():
            print('\n', '=' * 10, model_name, '=' * 10)
            early_stopping = utils.EarlyStopping(patience=config.early_stopping.patience, threshold=config.early_stopping.threshold)
            log, y_test_predicted = train_NN(MODEL_REGISTRY[model_name], params, task, train_loader, test_loader, val_loader, 
                        Adam, BCEWithLogitsLoss, config.general.epochs, verbose=True, scheduler_class=StepLR, early_stopping=early_stopping)
            agg_metrics = utils.metrics_to_string('validation', log['val loss'][-1], log['val accuracy'][-1], log['val precision'][-1], log['val recal'][-1], log['val f1'][-1])

            print('Train log\n', agg_metrics)
            if save_predictions:
                utils.save_predictions(y_test_predicted, test_indexes, f'objects/{task}/{model_name}', column_names=['PassengerId', config.targets.titanic])

    elif task == 'houses':

        for model_name, params in config.classic_ml_models.regression.items():
            train_metrics, y_test = train_classic_ml(MODEL_REGISTRY[model_name], params, task, **data)

            print('\n', '=' * 10, model_name, '=' * 10)
            agg_train = utils.aggregate_cv_metrics(train_metrics)

            print('Train log\n', agg_train)
            if save_predictions:
                utils.save_predictions(y_test, test_indexes, f'objects/{task}/{model_name}', column_names=['Id', config.targets.houses])

        for ensemble_name, params in config.ensembles.regression.items():
            estimators = [(model_name, MODEL_REGISTRY[model_name](**params)) for model_name, params in config.classic_ml_models.regression.items()]

            ensemble_params = dict(params.copy())
            ensemble_params['estimators'] = estimators
            if ensemble_name == 'StackingRegressor':
                ensemble_params['final_estimator'] = MODEL_REGISTRY[config.ensembles.stacking_regression_final_estimator]()
            train_metrics, y_test = train_classic_ml(MODEL_REGISTRY[ensemble_name], ensemble_params, task, **data)

            print('\n', '=' * 10, ensemble_name, '=' * 10)
            agg_train = utils.aggregate_cv_metrics(train_metrics)

            print('Train log\n', agg_train)
            if save_predictions:
                utils.save_predictions(y_test, test_indexes, f'objects/{task}/{ensemble_name}', column_names=['Id', config.targets.houses])

        for model_name, params in config.nn_models.regression.items():
            print('\n', '=' * 10, model_name, '=' * 10)
            early_stopping = utils.EarlyStopping(patience=config.early_stopping.patience, threshold=config.early_stopping.threshold)
            log, y_test_predicted = train_NN(MODEL_REGISTRY[model_name], params, task, train_loader, test_loader, val_loader, 
                        Adam, L1Loss, config.general.epochs, verbose=True, scheduler_class=StepLR, early_stopping=early_stopping)
            agg_metrics = utils.regression_metrics_to_string('validation', log['val loss'][-1], log['val mse'][-1], log['val mae'][-1], log['val r2'][-1], log['val rmse'][-1])

            print('Train log\n', agg_metrics)
            if save_predictions:
                utils.save_predictions(y_test_predicted, test_indexes, f'objects/{task}/{model_name}', column_names=['Id', config.targets.houses])