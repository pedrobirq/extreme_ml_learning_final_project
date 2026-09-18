import utils
from train_functions import run
from objects.datasets import TitanicDatasetPrepare, HousesDatasetPrepare
from config import config


def main():
    # print('~' * 15, "TITANIC DATASET", '~' * 15)
    # titanic_train_preparer = TitanicDatasetPrepare(config.paths.titanic_train)
    # titanic_test_preparer = TitanicDatasetPrepare(config.paths.titanic_test)

    # titanic_test_indexes = titanic_test_preparer.PassengerId

    # X_t_train, y_t_train = titanic_train_preparer.to_xy()
    # X_t_test = titanic_test_preparer.to_xy()

    # titanic_data = {'X_train': X_t_train, 'X_test': X_t_test, 'y_train': y_t_train}
    # run('titanic', titanic_data, titanic_test_indexes)
    # run('titanic', titanic_train_preparer, titanic_test_preparer, config.general.save_predictions)

    print('~' * 15, "CALIFORNIA HOUSES DATASET", '~' * 15)
    houses_train_preparer = HousesDatasetPrepare(config.paths.houses_train)
    houses_test_preparer = HousesDatasetPrepare(config.paths.houses_test)

    # houses_test_indexes = houses_test_preparer.Id

    # X_h_train, y_h_train = houses_train_preparer.to_xy()
    # X_h_test = houses_test_preparer.to_xy()

    # houses_data = {'X_train': X_h_train, 'X_test': X_h_test, 'y_train': y_h_train}
    run('houses', houses_train_preparer, houses_test_preparer, config.general.save_predictions)




if __name__ == '__main__':
    main()