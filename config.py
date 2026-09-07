from omegaconf import OmegaConf

conf = {
    'general': {
        'project_name': 'Titanic and Houses',
        'data_installed': True,

    },
    'paths': {
        'titanic_targets': 'objects/titanic/gender_submission.csv',
        'titanic_train': 'objects/titanic/train.csv',
        'titanic_test': 'objects/titanic/test.csv',
    }
}

config = OmegaConf.create(conf)