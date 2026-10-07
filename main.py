from src.datasets.titanic import TitanicDatasetPrepare
from configs.config_titanic import config


def main():
    # Data preprocessing
    dataset_preparer = TitanicDatasetPrepare(config.paths.train)
    df = dataset_preparer.prepare_dataset(columns_to_drop=config.drop_features, 
                                          OHE_cat_features=config.cat_features, 
                                          num_features=config.num_features,
                                          )
    X, y= dataset_preparer.to_xy(df)
    print(X.shape, y.shape)



if __name__ == '__main__':
    main()