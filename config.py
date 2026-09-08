from omegaconf import OmegaConf

conf = {
    'general': {
        'project_name': 'Titanic and Houses',
        'data_installed': True,

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
}

config = OmegaConf.create(conf)