from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, VotingRegressor, StackingRegressor

from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

from .dnn_models import SimpNN, MoreLayersNN, ImprovedNN

from torch.nn import BCEWithLogitsLoss, L1Loss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

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