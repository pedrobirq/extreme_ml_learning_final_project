import utils
from train_functions import run
from objects.datasets import TitanicDatasetPrepare, HousesDatasetPrepare
from config import config


def main():
    print('~' * 15, "TITANIC DATASET", '~' * 15)
    titanic_train_preparer = TitanicDatasetPrepare(config.paths.titanic_train)
    titanic_test_preparer = TitanicDatasetPrepare(config.paths.titanic_test)

    run('titanic', titanic_train_preparer, titanic_test_preparer, config.general.save_predictions)

    print('~' * 15, "CALIFORNIA HOUSES DATASET", '~' * 15)
    houses_train_preparer = HousesDatasetPrepare(config.paths.houses_train)
    houses_test_preparer = HousesDatasetPrepare(config.paths.houses_test)

    run('houses', houses_train_preparer, houses_test_preparer, config.general.save_predictions)


if __name__ == '__main__':
    main()