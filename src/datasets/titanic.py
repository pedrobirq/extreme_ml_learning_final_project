import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from configs.config_titanic import config
from src.utils import make_one_hot_encoding, make_min_max_scaling, make_standard_scaling

import torch
from torch.utils.data import Dataset



class TitanicDatasetPrepare:
    """ Preprocessing class for the titanic dataset """
    def __init__(self, path: str, is_train=True):
        self.df = pd.read_csv(path)
        self.is_train = is_train
        if self.is_train:
            self.statistics = dict()

    def _prepare_num_features(self, columns_to_prepare: list, df: pd.DataFrame, statistics=None, scaler='minmax'):

        prep_df = df.copy()
        for col in columns_to_prepare:
            if scaler == 'minmax': 
                encoded_col = make_min_max_scaling(prep_df[col])
            elif scaler == 'standard':
                if self.is_train:
                    encoded_col, mean, std = make_standard_scaling(prep_df[col], requires_statistics=True)
                    self.statistics[f"{col}_mean"] = mean
                    self.statistics[f"{col}_std"] = std
                else:
                    mean = statistics[f"{col}_mean"]
                    std = statistics[f"{col}_std"]
                    encoded_col = make_standard_scaling(prep_df[col], mean=mean, std=std)

            prep_df[col] = encoded_col

        return prep_df

    def _fill_nulls(self, df: pd.DataFrame):
        """
        Initials parsing from 'Name' column
        """
        df_no_nulls = df.copy()
        initials = df_no_nulls['Name'].str.extract('([A-Za-z]+)\.').squeeze()

        unique_initials = initials.unique().tolist()
        initials_to_replace = []
        for unique_initial in unique_initials:
            if unique_initial in ['Mlle', 'Mme', 'Ms', 'Dr', 'Major', 'Lady', 'Capt', 'Sir', 'Don', 'Dona']:
                if unique_initial in ['Mlle', 'Mme', 'Ms']:
                    initials_to_replace.append('Miss')
                elif unique_initial in ['Dr', 'Major', 'Capt', 'Sir', 'Don']:   
                    initials_to_replace.append('Mr')
                else:
                    initials_to_replace.append('Mrs')
            elif unique_initial in ['Mr', 'Mrs', 'Miss', 'Master']:
                initials_to_replace.append(unique_initial)
            else:
                initials_to_replace.append('Other')

        processed_initials = initials.replace(unique_initials, initials_to_replace)
        df_no_nulls['Initial'] = processed_initials

        mean_age_by_initial = df_no_nulls.groupby('Initial')['Age'].mean()

        for initial in mean_age_by_initial.keys():
            df_no_nulls.loc[(df_no_nulls['Age'].isnull()) & (df_no_nulls['Initial'] == initial), 'Age'] = mean_age_by_initial[initial]

        if self.is_train:
            df_no_nulls['Embarked'] = df_no_nulls['Embarked'].fillna('S')
        else:
            df_no_nulls.loc[(df_no_nulls['Fare'].isnull()), 'Fare'] = df_no_nulls.groupby('Pclass')['Fare'].mean()[3]

        return df_no_nulls

    def _make_OHE(self, columns_to_encode: list, df: pd.DataFrame, drop_first=True):

        prep_df = df.copy()
        encoded_columns = []
        for column in columns_to_encode:
            encoded_columns.append(make_one_hot_encoding(prep_df[column], drop_first=drop_first))

        ohe_df = pd.concat(encoded_columns, axis=1)

        # train/test sync
        if self.is_train:
            self.statistics['ohe_columns'] = list(ohe_df.columns)
        else:
            expected_ohe_cols = self.statistics['ohe_columns']
            ohe_df = ohe_df.reindex(columns=expected_ohe_cols, fill_value=0)

        prep_df = pd.concat([prep_df, ohe_df], axis=1)

        return prep_df.drop(columns=columns_to_encode)


        

    # def prepare_dataset(self, statistics=None):
    #     self._clean_nulls()
    #     self.df['Family_size'] = self.df['Parch'] + self.df['SibSp']
    #     self._prepare_cat_features()
    #     self._prepare_num_features(statistics)
    #     self.df['Sex'] = self.df['Sex'].map({'male': 0, 'female': 1})
    #     self.PassengerId = self.df['PassengerId']
    #     self.df = self.df.drop(columns=['Parch', 'SibSp', 'Name', 'Ticket', 'Cabin', 'PassengerId', 'Pclass', 'Embarked', 'Initial'])

    #     return self.df

    def prepare_dataset(self, 
                        statistics=None, 
                        drop_duplicates=True, 
                        columns_to_drop=None, 
                        fill_nulls=True,
                        make_family_size_column=True,
                        OHE_cat_features=None,
                        num_features=None,
                        scaler='minmax'):
        prep_df = self.df.copy()

        prep_df['Sex'] = prep_df['Sex'].map({'male': 0, 'female': 1})
        self.PassengerId = prep_df['PassengerId']

        if drop_duplicates:
            prep_df = prep_df.drop_duplicates()

        if fill_nulls:
            prep_df = self._fill_nulls(prep_df)

        if make_family_size_column:
            prep_df['Family_size'] = prep_df['Parch'] + prep_df['SibSp']

        if OHE_cat_features is not None:
            prep_df = self._make_OHE(OHE_cat_features, prep_df)

        if num_features is not None:
            if not self.is_train and statistics is not None:
                prep_df = self._prepare_num_features(num_features, prep_df, scaler=scaler, statistics=statistics)
            else:
                prep_df = self._prepare_num_features(num_features, prep_df, scaler=scaler)

        if columns_to_drop is not None:
            prep_df = prep_df.drop(columns=columns_to_drop)

        return prep_df


    # def to_xy(self):
    #     if 'Name' in self.df.columns:
    #         self.prepare_dataset()
    #     if self.is_train:
    #         y = self.df['Survived'].to_numpy()
    #         X = self.df.drop(columns=['Survived']).to_numpy()
    #         return X, y
    #     else:
    #         X = self.df.to_numpy()
    #         return X


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