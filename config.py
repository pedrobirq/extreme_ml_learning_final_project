from omegaconf import OmegaConf

conf = {
    'general': {
        'project_name': 'Titanic and Houses',
        'data_installed': True,
        'random_state': 52,
        'test_size': 0.2,
        
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
    }
}

config = OmegaConf.create(conf)