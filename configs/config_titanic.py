from omegaconf import OmegaConf

conf = {
    'general': {
        'project_name': 'Titanic',
        'data_installed': True,
        'random_state': 52,
        'test_size': 0.2,
        'learning_rate': 0.1,
        'device': 'cpu',
        'batch_size': 16,
        'epochs': 150,
        'save_predictions': False,
        'target': 'Survived',
    },
    'paths': {
        'train': 'data/titanic/train.csv',
        'test': 'data/titanic/test.csv',
    },
    'cat_features': ['Pclass', 'Embarked', 'Initial'],
    'num_features': ['Age', 'Fare', 'Family_size'],
    'drop_features': ['Name', 'PassengerId', 'Parch', 'SibSp', 'Ticket', 'Cabin',  'Pclass'],
    'cv': {
        'k_folds': 5,
        'shuffle': True,
    },
    'classic_ml_models': {
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
    'ensembles': {
        'VotingClassifier': {
            'voting': 'hard'
        },
        'StackingClassifier': {
            'stack_method': 'auto'
        },
        'stacking_classification_final_estimator': 'LogisticRegression',
        'stacking_regression_final_estimator': 'LinearRegression',  
    },
    'nn_models': {
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