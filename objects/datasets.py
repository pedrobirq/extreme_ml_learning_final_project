import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import config
import utils

import torch
from torch.utils.data import Dataset


class TitanicDatasetPrepare:
    """ Preprocessing class for the titanic dataset """
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
        self.PassengerId = self.df['PassengerId']
        self.df = self.df.drop(columns=['Parch', 'SibSp', 'Name', 'Ticket', 'Cabin', 'PassengerId', 'Pclass', 'Embarked', 'Initial'])

        return self.df

    def to_xy(self):
        if 'Name' in self.df.columns:
            self.prepare_dataset()
        if self.is_train:
            y = self.df['Survived'].to_numpy()
            X = self.df.drop(columns=['Survived']).to_numpy()
            return X, y
        else:
            X = self.df.to_numpy()
            return X


class TitanicDatasetPrepareNN(Dataset):
    """ dNNs wrapper class for the titanic dataset """
    def __init__(self, df: pd.DataFrame):
        if 'Survived' in df.columns:
            self.is_train = True
            self.y = torch.tensor(df['Survived'].to_numpy(), dtype=torch.float32).unsqueeze(1)
            self.X = torch.tensor(df.drop(columns=['Survived']).to_numpy(), dtype=torch.float32)
        else:
            self.is_train = False
            self.X = torch.tensor(df.to_numpy(), dtype=torch.float32)


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        if self.is_train:
            return self.X[index], self.y[index]
        else:
            return self.X[index]


class HousesDatasetPrepare:
    """ Preprocessing class for the houses dataset"""
    def __init__(self, path: str):
        self.df = pd.read_csv(path)
        self.is_train = 'train' in path
        if self.is_train:
            self.statistics = dict()
        
    def _prepare_cat_features(self, statistics=None):
        # Ordinal relationship
        quality_related_cols = ["ExterCond", "ExterQual", "BsmtQual", "BsmtCond", "HeatingQC", 
                                "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]
        basement_cols = ["BsmtFinType1", "BsmtFinType2"]

        quality_dict = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}
        basement_quality_dict = {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0}
        exposure_quality_dict = {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}
        air_dict = {"N": 0, "Y": 1}

        self.df[quality_related_cols] = self.df[quality_related_cols].map(quality_dict.get)
        self.df[basement_cols] = self.df[basement_cols].map(basement_quality_dict.get)
        self.df['BsmtExposure'] = self.df['BsmtExposure'].map(exposure_quality_dict.get)
        self.df['CentralAir'] = self.df['CentralAir'].map(air_dict.get)

        self.ord_columns = quality_related_cols + basement_cols + ['BsmtExposure', 'CentralAir']

        self.one_hot_columns = sorted(list(set(config.cat_features.houses) - set(self.ord_columns)))

        # OneHot encoding
        encoded_columns = []
        for col in self.one_hot_columns:
            encoded_col = utils.make_one_hot_encoding(self.df[col], drop_first=True)
            encoded_columns.append(encoded_col)
        
        ohe_df = pd.concat(encoded_columns, axis=1)

        # Synchronization of OHE columns between train and test
        if self.is_train:
            self.statistics['ohe_columns'] = list(ohe_df.columns)
        else:
            expected_ohe_cols = statistics['ohe_columns']
            ohe_df = ohe_df.reindex(columns=expected_ohe_cols, fill_value=0)

        self.df = pd.concat([self.df, ohe_df], axis=1)

    def _prepare_num_features(self, statistics=None):
        need_to_scale = config.num_features.houses + self.ord_columns
        for col in need_to_scale:
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
        cat_cols = list(config.cat_features.houses)
        num_cols = list(config.num_features.houses)
    
        self.df[cat_cols] = self.df[cat_cols].fillna('None')
        self.df[num_cols] = self.df[num_cols].fillna(0)

    def prepare_dataset(self, statistics=None):
        self._clean_nulls()
        self._prepare_cat_features(statistics)
        self._prepare_num_features(statistics)
        self.Id = self.df['Id']
        self.df = self.df.drop(columns=self.one_hot_columns + ['Id'])

        if self.is_train:
            feature_cols = [c for c in self.df.columns if c != config.targets.houses]
            self.statistics['feature_columns'] = feature_cols
            self.df[config.targets.houses] = np.log1p(self.df[config.targets.houses])
        else:
            expected_features = statistics['feature_columns']
            self.df = self.df.reindex(columns=expected_features)

        return self.df

    def to_xy(self, statistics=None):
        if 'Id' in self.df.columns:
            self.prepare_dataset(statistics=statistics)
        if self.is_train:
            y = self.df[config.targets.houses].to_numpy()
            X = self.df.drop(columns=[config.targets.houses]).to_numpy()
            return X, y
        else:
            X = self.df.to_numpy()
            return X


class HousesDatasetPrepareNN(Dataset):
    """ dNNs wrapper class for the houses dataset """
    def __init__(self, df: pd.DataFrame):
        if config.targets.houses in df.columns:
            self.is_train = True
            self.y = torch.tensor(df[config.targets.houses].to_numpy(), dtype=torch.float32).unsqueeze(1)
            self.X = torch.tensor(df.drop(columns=[config.targets.houses]).to_numpy(), dtype=torch.float32)
        else:
            self.is_train = False
            self.X = torch.tensor(df.to_numpy(), dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        if self.is_train:
            return self.X[index], self.y[index]
        else:
            return self.X[index]