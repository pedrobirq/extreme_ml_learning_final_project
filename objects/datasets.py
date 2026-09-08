import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import config
import utils


class TitanicDatasetPrepare:
    def __init__(self, path: str):
        self.df = pd.read_csv(path)
        self.is_train = 'train' in path
        if self.is_train:
            self.statistics = dict()
        
    def _prepare_cat_features(self):
        encoded_columns = []
        for col in config.cat_features.titanic:
            encoded_col = utils.make_one_hot_encoding(self.df[col], drop_first=True)
            encoded_columns.append(encoded_col)
        self.df = pd.concat([self.df, *encoded_columns], axis=1)

    def _prepare_num_features(self, statistics=None):
        for col in config.num_features.titanic:
            if self.is_train:
                encoded_col, mean, std = utils.make_standard_scaling(self.df[col], requires_statistics=True)
                self.df[col] = encoded_col
                self.statistics[f"{col}_mean"] = mean
                self.statistics[f"{col}_std"] = std
            else:
                mean = statistics[f"{col}_mean"]
                std = statistics[f"{col}_std"]
                encoded_col = utils.make_standard_scaling(self.df[col], mean=mean, std=std)
                self.df[col] = encoded_col

            
    def _clean_nulls(self):
        initial = utils.titanic_make_initials_column(self.df['Name'])
        self.df['Initial'] = initial
        self.df = utils.titanic_fill_nulls(self.df, self.is_train)

    def prepare_dataset(self, statistics=None):
        self._clean_nulls()
        self.df['Family_size'] = self.df['Parch'] + self.df['SibSp']
        self._prepare_cat_features()
        self._prepare_num_features(statistics)
        self.df['Sex'] = self.df['Sex'].map({'male': 0, 'female': 1})
        # print(self.df.head())
        self.df = self.df.drop(columns=['Parch', 'SibSp', 'Name', 'Ticket', 'Cabin', 'PassengerId', 'Pclass', 'Embarked', 'Initial'])

        if self.is_train:
            y = self.df['Survived']
            X = self.df.drop(columns=['Survived'])
        else:
            X = self.df

        return self.df

    def to_xy(self):
        if 'Name' in self.df.columns:
            raise ValueError("Dataset is not prepared yet. Please call prepare_dataset() first.")
        if self.is_train:
            y = self.df['Survived'].to_numpy()
            X = self.df.drop(columns=['Survived']).to_numpy()
            return X, y
        else:
            X = self.df.to_numpy()
            return X






