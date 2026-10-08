import pandas as pd
import numpy as np
from typing import Union
# import matplotlib.pyplot as plt

from configs.config_titanic import config
from src.utils import make_one_hot_encoding, make_min_max_scaling, make_standard_scaling

import torch
from torch.utils.data import Dataset



class TitanicDatasetPrepare:
    """ Preprocessing class for the titanic dataset """

    def __init__(self, data: Union[str, pd.DataFrame], is_train=True, statistics=None):
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        else:
            self.df = data.copy()
        self.is_train = is_train
        if self.is_train:
            self.statistics = dict()
        else:
            if statistics is not None:
                self.statistics = statistics
            else:
                print('Provide train statistics first')


    def _prepare_num_features(self, columns_to_prepare: list, df: pd.DataFrame, scaler='minmax'):

        prep_df = df.copy()
        for col in columns_to_prepare:

            if scaler == 'minmax': 
                if self.is_train:
                    col_min = prep_df[col].min()
                    col_max = prep_df[col].max()
                    self.statistics[f"{col}_min"] = col_min
                    self.statistics[f"{col}_max"] = col_max
                else:
                    col_min = self.statistics[f"{col}_min"]
                    col_max = self.statistics[f"{col}_max"]

                # encoded_col = make_min_max_scaling(prep_df[col])
                # Защита от деления на 0 при константном признаке
                denom = (col_max - col_min) if (col_max - col_min) != 0 else 1.0
                prep_df[col] = (prep_df[col] - col_min) / denom

            elif scaler == 'standard':
                if self.is_train:
                    prep_df[col], mean, std = make_standard_scaling(prep_df[col], requires_statistics=True)
                    self.statistics[f"{col}_mean"] = mean
                    self.statistics[f"{col}_std"] = std
                else:
                    mean = self.statistics[f"{col}_mean"]
                    std = self.statistics[f"{col}_std"]
                    prep_df[col] = make_standard_scaling(prep_df[col], mean=mean, std=std)


        return prep_df
    

    def _fill_nulls(self, df: pd.DataFrame):
        """
        Initials parsing from 'Name' column
        """
        df_no_nulls = df.copy()
        initials = df_no_nulls['Name'].str.extract('([A-Za-z]+)\.').squeeze()

        title_mapping = {
            'Mlle': 'Miss', 'Mme': 'Miss', 'Ms': 'Miss',
            'Dr': 'Mr', 'Major': 'Mr', 'Capt': 'Mr', 'Sir': 'Mr', 'Don': 'Mr',
            'Lady': 'Mrs', 'Dona': 'Mrs',
            'Mr': 'Mr', 'Mrs': 'Mrs', 'Miss': 'Miss', 'Master': 'Master'
        }
        df_no_nulls['Initial'] = initials.map(lambda x: title_mapping.get(x, 'Other'))

        if self.is_train:
            # Статистики возраста
            self.statistics['mean_age_by_initial'] = df_no_nulls.groupby('Initial')['Age'].mean().to_dict()
            self.statistics['global_mean_age'] = df_no_nulls['Age'].mean()
            
            # Мода Embarked
            embarked_mode = df_no_nulls['Embarked'].mode()
            self.statistics['embarked_mode'] = embarked_mode.iloc[0] if not embarked_mode.empty else 'S'
            
            # Средняя стоимость билета по Pclass
            self.statistics['mean_fare_by_pclass'] = df_no_nulls.groupby('Pclass')['Fare'].mean().to_dict()
            self.statistics['global_mean_fare'] = df_no_nulls['Fare'].mean()

        # Применение статистик к текущему фолду
        # 1. Возраст
        mean_ages = self.statistics['mean_age_by_initial']
        global_age = self.statistics.get('global_mean_age', 28.0)
        for init, avg_age in mean_ages.items():
            mask = (df_no_nulls['Age'].isnull()) & (df_no_nulls['Initial'] == init)
            df_no_nulls.loc[mask, 'Age'] = avg_age
        df_no_nulls['Age'] = df_no_nulls['Age'].fillna(global_age)

        # 2. Embarked
        df_no_nulls['Embarked'] = df_no_nulls['Embarked'].fillna(self.statistics['embarked_mode'])

        # 3. Fare
        mean_fares = self.statistics['mean_fare_by_pclass']
        global_fare = self.statistics.get('global_mean_fare', 32.0)
        for pclass, avg_fare in mean_fares.items():
            mask = (df_no_nulls['Fare'].isnull()) & (df_no_nulls['Pclass'] == pclass)
            df_no_nulls.loc[mask, 'Fare'] = avg_fare
        df_no_nulls['Fare'] = df_no_nulls['Fare'].fillna(global_fare)

        return df_no_nulls

    def _make_OHE(self, columns_to_encode: list, df: pd.DataFrame, drop_first=True):
        prep_df = df.copy()

        ohe_df = pd.get_dummies(prep_df[columns_to_encode], drop_first=drop_first, dtype=float)

        # train/val sync
        if self.is_train:
            self.statistics['ohe_columns'] = list(ohe_df.columns)
        else:
            expected_ohe_cols = self.statistics['ohe_columns']
            ohe_df = ohe_df.reindex(columns=expected_ohe_cols, fill_value=0)

        prep_df = pd.concat([prep_df.drop(columns=columns_to_encode), ohe_df], axis=1)

        return prep_df


    def prepare_dataset(self,
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
            prep_df = self._prepare_num_features(num_features, prep_df, scaler=scaler)

        if columns_to_drop is not None:
            cols_to_remove = [c for c in columns_to_drop if c in prep_df.columns]
            prep_df = prep_df.drop(columns=cols_to_remove)

        return prep_df


    def to_xy(self, prep_df):
        if config.general.target in prep_df.columns:
            y = prep_df[config.general.target].to_numpy()
            X = prep_df.drop(columns=[config.general.target]).to_numpy()
            return X, y
        else:
            X = prep_df.to_numpy()
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