from src.datasets.titanic import TitanicDatasetPrepare
from configs.config_titanic import config


def main():
    # Data preprocessing
    dataset_preparer = TitanicDatasetPrepare(config.paths.train)
    df = dataset_preparer.prepare_dataset()
    print(df.isna().sum())



if __name__ == '__main__':
    main()