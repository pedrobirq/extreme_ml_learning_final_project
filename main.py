from src.train_functions import train_classification, run_all_models, make_test_predictions, objective, make_folds, run_optuna
from src import utils
from src.models.registry import MODEL_PARAMS
from configs.config_titanic import config
import optuna
import pandas as pd


def main():
    MODEL_NAME = 'LogisticRegression'
    is_nn = False
    
    nn_attributes = {'scheduler_class': 'StepLR',
                    'loss_class': 'BCEWithLogitsLoss',
                    'optimizer_class': 'Adam'} if is_nn else None

    folds, fold_statistics = make_folds(config.preprocessing_params, is_nn=is_nn)

    cv_metrics, models = train_classification( MODEL_NAME, 
                                                MODEL_PARAMS[MODEL_NAME], 
                                                folds,
                                                nn_attributes=nn_attributes 
                                                )
    
    cv_statistics = utils.calc_cv_statistics(cv_metrics)
    make_test_predictions(models, fold_statistics, config.preprocessing_params, MODEL_NAME + '_with_ElasticNet', is_nn=is_nn)
    print(cv_statistics)

    # ~~~~~ OPTUNA INVOLVE ~~~~~~
    
    # run_optuna(model_name=MODEL_NAME,
    #            target_metric='val_f1',
    #            folds=folds)

if __name__ == '__main__':
    main()