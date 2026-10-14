from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, VotingRegressor, StackingRegressor

from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

from .dnn_models import SimpNN, MoreLayersNN, ImprovedNN
from configs.config_titanic import config

from torch.nn import BCEWithLogitsLoss, L1Loss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from typing import Any, Dict, Optional
import optuna

CLASSIC_ML_MODEL_REGISTRY = {
    'LogisticRegression': LogisticRegression,
    'KNeighborsClassifier': KNeighborsClassifier,
    'DecisionTreeClassifier': DecisionTreeClassifier,
    'RandomForestClassifier': RandomForestClassifier,
    'VotingClassifier': VotingClassifier,
    'StackingClassifier': StackingClassifier,
    'CatBoostClassifier': CatBoostClassifier,
    'LGBMClassifier': LGBMClassifier,
    'XGBClassifier': XGBClassifier,
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


DL_REGISTRY = {
    'SimpNN': SimpNN,
    'MoreLayersNN': MoreLayersNN,
    'ImprovedNN': ImprovedNN,
}

NN_ATTRIBUTES = {
    # Losses
    'BCEWithLogitsLoss': BCEWithLogitsLoss,
    'L1Loss': L1Loss,

    # Schedulers
    'StepLR': StepLR,
    'CosineAnnealingLR': CosineAnnealingLR,

    # Optimizers
    'Adam': Adam
}


# --- Logistic Regression ---
def get_logistic_regression_params(
    trial: Optional[optuna.Trial]
) -> Dict[str, Any]:
    base_params = {
        "max_iter": 1000,
        "random_state": config.general.random_state,
    }
    if trial is None:
        return base_params

    # Тонкость: выбор регуляризации и совместимого солвера
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])

    if penalty == "l1":
        solver = trial.suggest_categorical("solver_l1", ["liblinear", "saga"])
    elif penalty == "elasticnet":
        solver = "saga"
    else:  # l2
        solver = trial.suggest_categorical("solver_l2", ["lbfgs", "liblinear", "saga"])

    search_params = {
        "C": trial.suggest_float("C", 1e-4, 1e2, log=True),
        "penalty": penalty,
        "solver": solver,
    }

    if penalty == "elasticnet":
        search_params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.05, 0.95)

    return {**base_params, **search_params}


# --- CatBoostClassifier ---
def get_catboost_params(
    trial: Optional[optuna.Trial]
) -> Dict[str, Any]:
    base_params = {
        "loss_function": "Logloss",
        "eval_metric": "F1",
        "random_seed": config.general.random_state,
        "verbose": False,
    }
    if trial is None:
        return base_params

    bootstrap_type = trial.suggest_categorical(
        "bootstrap_type", ["Bayesian", "Bernoulli"]
    )

    search_params = {
        "iterations": trial.suggest_int("iterations", 300, 1200, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "random_strength": trial.suggest_float(
            "random_strength", 1e-3, 10.0, log=True
        ),
        "bootstrap_type": bootstrap_type,
    }

    if bootstrap_type == "Bayesian":
        search_params["bagging_temperature"] = trial.suggest_float(
            "bagging_temperature", 0.0, 1.0
        )
    elif bootstrap_type == "Bernoulli":
        search_params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

    return {**base_params, **search_params}


# --- XGBClassifier ---
def get_xgboost_params(
    trial: Optional[optuna.Trial]
) -> Dict[str, Any]:
    base_params = {
        "eval_metric": "logloss",
        "random_state": config.general.random_state,
        "n_jobs": -1,
    }
    if trial is None:
        return base_params

    search_params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    return {**base_params, **search_params}


# --- LGBMClassifier ---
def get_lgbm_params(
    trial: Optional[optuna.Trial]
) -> Dict[str, Any]:
    base_params = {
        "random_state": config.general.random_state,
        "verbose": -1,
        "n_jobs": -1,
    }
    if trial is None:
        return base_params

    search_params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        # Важно: subsample в LightGBM работает только если subsample_freq > 0
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    return {**base_params, **search_params}


# Единый реестр функций генерации параметров
PARAMETERS_REGISTRY = {
    "LogisticRegression": get_logistic_regression_params,
    "CatBoostClassifier": get_catboost_params,
    "XGBClassifier": get_xgboost_params,
    "LGBMClassifier": get_lgbm_params,
}