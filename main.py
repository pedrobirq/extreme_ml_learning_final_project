# from src.datasets.titanic import TitanicDatasetPrepare
from src.train_functions import train_classification, run_all_models, make_test_predictions
from src import utils
from configs.config_titanic import config
import pandas as pd


def main():
    # Data preprocessing
    # dataset_preparer = TitanicDatasetPrepare(config.paths.train)
    # df = dataset_preparer.prepare_dataset(columns_to_drop=config.drop_features, 
    #                                       OHE_cat_features=config.cat_features, 
    #                                       num_features=config.num_features,
    #                                       )
    # X, y = dataset_preparer.to_xy(df)

    # test_dataset_preparer = TitanicDatasetPrepare(config.paths.test, is_train=False, statistics=dataset_preparer.statistics)
    # test_df = test_dataset_preparer.prepare_dataset(columns_to_drop=config.drop_features,
    #                                                 OHE_cat_features=config.cat_features, 
    #                                                 num_features=config.num_features,
    #                                                 )
    # X_test = test_dataset_preparer.to_xy(test_df)
    # print(X_test.shape, X.shape)
    # print(df.head(), '\n\n\n', test_df.head())

    cv_metrics, models, fold_statistics = train_classification('SimpNN', 
                                                               config.nn_models.SimpNN, 
                                                               config.preprocessing_params,
                                                               config=config,
                                                               nn_attributes={'scheduler_class': 'StepLR',
                                                                              'loss_class': 'BCEWithLogitsLoss',
                                                                              'optimizer_class': 'Adam'})

    cv_statistics = utils.calc_cv_statistics(cv_metrics)
    # make_test_predictions(models, fold_statistics, config.preprocessing_params, config, 'Baseline_LogisticRegression')
    # print('~' * 10, 'LogisticRegression', '~' * 10)
    print(cv_statistics)
    # run_all_models('classic_ml_models', config.preprocessing_params, config)


if __name__ == '__main__':
    main()