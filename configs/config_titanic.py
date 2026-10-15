from omegaconf import OmegaConf

conf = {
    'general': {
        'project_name': 'Titanic',
        'data_installed': True,
        'random_state': 52,
        'test_size': 0.2,
        'learning_rate': 0.1,
        'device': 'cuda',
        'batch_size': 32,
        'epochs': 100,
        'save_predictions': False,
        'save_weights': True,
        'target': 'Survived',
    },
    'paths': {
        'train': 'data/titanic/train.csv',
        'test': 'data/titanic/test.csv',
        'submissions': 'submissions/titanic/'
    },
    'cat_features': ['Pclass', 'Embarked', 'Initial'],
    'num_features': ['Age', 'Fare', 'Family_size'],
    'preprocessing_params': {
        'columns_to_drop': ['Name', 'PassengerId', 'Parch', 'SibSp', 'Ticket', 'Cabin'],
        'OHE_cat_features': ['Pclass', 'Embarked', 'Initial'], 
        'num_features': ['Age', 'Fare', 'Family_size'],
    },
    'cv': {
        'k_folds': 5,
        'shuffle': True,
    },
    'classic_ml_models': {
        'LogisticRegression': {
            'C': 1.2877324848977032,
            'random_state': '${general.random_state}',
            # 'penalty': 'elasticnet',
            'l1_ratio': 0.07633822231326909,
            'max_iter': 1000,
            'solver': 'saga'
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
            'loss_function': 'Logloss', 
            'eval_metric': 'F1', 
            'random_seed': '${general.random_state}', 
            'verbose': False, 
            'bootstrap_type': 'Bayesian', 
            'iterations': 700, 
            'learning_rate': 0.02273501424836124, 
            'depth': 5, 
            'l2_leaf_reg': 2.2500560852033615, 
            'random_strength': 1.0274389459259032, 
            'bagging_temperature': 0.4834678819225746},
        'LGBMClassifier': {
            'random_state': '${general.random_state}',
            'verbose': -1, 
            'n_jobs': -1, 
            'n_estimators': 450, 
            'learning_rate': 0.09223630732225964, 
            'num_leaves': 237, 
            'max_depth': 4, 
            'min_child_samples': 44, 
            'subsample': 0.5571343990506431, 
            'subsample_freq': 1, 
            'colsample_bytree': 0.9801302473217693, 
            'reg_alpha': 1.877176178394129e-05, 
            'reg_lambda': 2.609503675904433e-05
        },
        'XGBClassifier': {
            'eval_metric': 'logloss', 
            'random_state': '${general.random_state}', 
            'n_jobs': -1, 
            'n_estimators': 450, 
            'learning_rate': 0.022379591615960075, 
            'max_depth': 10, 
            'min_child_weight': 2, 
            'subsample': 0.7395324749096183, 
            'colsample_bytree': 0.7374217679653282, 
            'gamma': 0.00025392580097712816, 
            'reg_alpha': 0.017726640689496854, 
            'reg_lambda': 0.0001048736994376153
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
            'input': 11,
            'output': 1,
            'hidden_size': 64
        },
        'MoreLayersNN': {
            'input': 11,
            'output': 1,
            'hidden_size1': 64,
            'hidden_size2': 32
        },
        'ImprovedNN': {
            'input': 11,
            'output': 1,
            'hidden_size1': 128,
            'hidden_size2': 64
        },
    },
    'schedulers': {
        'step_lr': {
                    'step_size': 5,
                    'gamma': 0.95
                },
        'cosine': {
            'T_max': '${general.epochs}',
            'eta_min': 1e-5
        }
    },
    'early_stopping': {
        'threshold': 1e-4,
        'patience': 15,
        'mode': 'min',
        'patience': 20,
        'threshold_mode': 'rel',
        'save_optimizer': True
    }
}

config = OmegaConf.create(conf)