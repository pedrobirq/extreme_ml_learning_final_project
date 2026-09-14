from train_functions import run
from objects.datasets import TitanicDatasetPrepare
from config import config


def main():
    print('~' * 15, "TITANIC DATASET", '~' * 15)
    titanic_train_preparer = TitanicDatasetPrepare(config.paths.titanic_train)
    titanic_train_preparer.prepare_dataset()
    titanic_statistics = titanic_train_preparer.statistics
    titanic_test_preparer = TitanicDatasetPrepare(config.paths.titanic_test)
    titanic_test_preparer.prepare_dataset(titanic_statistics)

    titanic_test_indexes = titanic_test_preparer.PassengerId

    X_t_train, y_t_train = titanic_train_preparer.to_xy()
    X_t_test = titanic_test_preparer.to_xy()

    titanic_data = {'X_train': X_t_train, 'X_test': X_t_test, 'y_train': y_t_train}
    run('titanic', titanic_data, titanic_test_indexes)


if __name__ == '__main__':
    main()