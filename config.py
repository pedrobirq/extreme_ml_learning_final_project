from omegaconf import OmegaConf
import torch

conf = {
    'general': {
        'project_name': 'Titanic and Houses',
        'data_installed': True,
        'random_state': 52,
        'test_size': 0.2,
        'learning_rate': 0.1,
        'device': 'cuda',
        'batch_size': 16,
        'epochs': 150,
        'save_predictions': False,
    },
    'paths': {
        'titanic_train': 'objects/titanic/train.csv',
        'titanic_test': 'objects/titanic/test.csv',
        'houses_train': 'objects/houses/train.csv',
        'houses_test': 'objects/houses/test.csv'
    },
    'cat_features': {
        'titanic': ['Pclass', 'Embarked', 'Initial'],
        'houses': ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
       'SaleType', 'SaleCondition']
    },
    'num_features': {
        'titanic': ['Age', 'Fare', 'Family_size'],
        'houses': ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
       'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
       'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
       'MoSold', 'YrSold']
    },
    'targets': {
        'titanic': 'Survived',
        'houses': 'SalePrice'
    },
    'cv': {
        'k_folds': 5,
        'shuffle': True,
    },
    'classic_ml_models': {
        'classification': {
            'LogisticRegression': {
                'C': 2.82,
                'random_state': '${general.random_state}'
            },
            'KNeighborsClassifier': {
                'n_neighbors': 5
            },
            'DecisionTreeClassifier': {
                'criterion': 'log_loss',
                'min_samples_leaf': 3,
                'min_samples_split': 10,
                'random_state': '${general.random_state}'
            },
            'RandomForestClassifier': {
                'criterion': 'log_loss',
                'min_samples_leaf': 3,
                'min_samples_split': 10,
                'n_estimators': 30,
                'random_state': '${general.random_state}'
            },
            'CatBoostClassifier': {
                'iterations': 100,
                'depth': 4,
                'learning_rate': '${general.learning_rate}',
                'loss_function': 'Logloss',
                'verbose': False,
                'l2_leaf_reg': 5,
                'bagging_temperature': 1.0,
                'random_strength': 1.0,
                'random_state': '${general.random_state}'
            },
            'LGBMClassifier': {
                'metric': 'binary_logloss',
                'verbose': -1,
                'num_round': 100,
                'max_depth': 4,
                'num_leaves': 15,
                'min_child_samples': 15,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': '${general.random_state}'
            },
            'XGBClassifier': {
                'n_estimators': 100,
                'max_depth': 4,
                'learning_rate': '${general.learning_rate}',
                'objective': 'binary:logistic',
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': '${general.random_state}'
            },
        },
        'regression': {
            'LinearRegression': {
            },
            'Lasso': {
                'alpha': 1e-3,
                'max_iter': 15000,
                'random_state': '${general.random_state}'
            },
            'Ridge': {
                'alpha': 20.35,
                'solver': 'svd',
                'random_state': '${general.random_state}'
            },
            'ElasticNet': {
                'alpha': 1,
                'l1_ratio': 0.05,
                'random_state': '${general.random_state}'
            },
            'KNeighborsRegressor': {
                'algorithm': 'ball_tree',
                'n_neighbors': 9
            },
            'CatBoostRegressor': {
                'iterations': 1000,
                'depth': 3,
                'learning_rate': '${general.learning_rate}',
                'loss_function': 'RMSE',
                'verbose': False,
                'l2_leaf_reg': 5,
                'bagging_temperature': 1.0,
                'random_strength': 1.0,
                'random_state': '${general.random_state}'
            },
            'LGBMRegressor': {
                'metric': 'RMSE',
                'verbose': -1,
                'num_round': 1000,
                'max_depth': 3,
                'num_leaves': 15,
                'min_child_samples': 15,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': '${general.random_state}'
            },
            'XGBRegressor': {
                'n_estimators': 1000,
                'max_depth': 3,
                'learning_rate': '${general.learning_rate}',
                'objective': 'reg:squarederror',
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': '${general.random_state}'
            }
        }
    },
    'ensembles': {
        'classification': {
            'VotingClassifier': {
                'voting': 'hard'
            },
            'StackingClassifier': {
                'stack_method': 'auto'
            }
        },
        'regression': {
            'VotingRegressor': {

            },
            'StackingRegressor': {

            },
        },
        'stacking_classification_final_estimator': 'LogisticRegression',
        'stacking_regression_final_estimator': 'LinearRegression'
    },
    'nn_models': {
        'classification': {
            'SimpNN': {
                'input': 12,
                'output': 1,
                'hidden_size': 64
            },
            'MoreLayersNN': {
                'input': 12,
                'output': 1,
                'hidden_size1': 64,
                'hidden_size2': 32
            },
            'ImprovedNN': {
                'input': 12,
                'output': 1,
                'hidden_size1': 128,
                'hidden_size2': 64
            },
        },
        'regression': {
            'SimpNN': {
                'input': 217,
                'output': 1,
                'hidden_size': 64
            },
            'MoreLayersNN': {
                'input': 217,
                'output': 1,
                'hidden_size1': 64,
                'hidden_size2': 32
            },
            'ImprovedNN': {
                'input': 217,
                'output': 1,
                'hidden_size1': 64,
                'hidden_size2': 32
            },
        }
    },
    'schedulers': {
        'step_lr': {
                    'step_size': 5,
                    'gamma': 0.95
                }
    },
    'early_stopping': {
        'threshold': 1e-4,
        'patience': 15
    }
}

config = OmegaConf.create(conf)