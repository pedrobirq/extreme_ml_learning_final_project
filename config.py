from omegaconf import OmegaConf

conf = {
    'general': {
        'project_name': 'Titanic and Houses',
        'data_installed': True,
        'random_state': 52,
        'test_size': 0.2,
        'learning_rate': 0.1,
    },
    'paths': {
        'titanic_train': 'objects/titanic/train.csv',
        'titanic_test': 'objects/titanic/test.csv',
    },
    'cat_features': {
        'titanic': ['Pclass', 'Embarked', 'Initial'],
    },
    'num_features': {
        'titanic': ['Age', 'Fare', 'Family_size'],
    },
    'cv': {
        'k_forlds': 5,
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
            }
        },
    }
}

config = OmegaConf.create(conf)