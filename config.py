from omegaconf import OmegaConf

config = {
    'general': {
        'project_name': 'Titanic and Houses'
    }
}

config = OmegaConf.create(config)
