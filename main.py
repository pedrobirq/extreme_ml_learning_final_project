# from src.datasets.titanic import TitanicDatasetPrepare
from src.train_functions import train_classification, run_all_models, make_test_predictions, objective
from src import utils
from configs.config_titanic import config
import optuna
import pandas as pd


def main():
    nn_attributes={'scheduler_class': 'CosineAnnealingLR',
                    'loss_class': 'BCEWithLogitsLoss',
                    'optimizer_class': 'Adam'}

    cv_metrics, models, fold_statistics = train_classification('SimpNN', 
                                                               config.nn_models.SimpNN, 
                                                               config.preprocessing_params,
                                                               nn_attributes=nn_attributes
                                                               )

    cv_statistics = utils.calc_cv_statistics(cv_metrics)
    make_test_predictions(models, fold_statistics, config.preprocessing_params, config, 'Test', nn=True)
    print(cv_statistics)

    # ~~~~~ OPTUNA INVOLVE ~~~~~~

    # sampler = optuna.samplers.TPESampler(seed=config.general.random_state)

    # study = optuna.create_study(
    #     direction="maximize",  # максимизируем f1
    #     sampler=sampler,
    #     study_name="catboost_f1_optimization",
    # )

    # # n_trials: количество итераций; timeout: лимит в секундах
    # study.optimize(objective, n_trials=30, timeout=600, show_progress_bar=True)

    # print(f"Лучший результат f1: {study.best_value:.5f}")
    # print("Лучшие параметры:")
    # for key, value in study.best_params.items():
    #     print(f"  {key}: {value}")

if __name__ == '__main__':
    main()