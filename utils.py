import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Preprocessing


def make_one_hot_encoding(series: pd.Series, drop_first=False) -> pd.DataFrame:
    """
    Performs OneHotEncoding on a Series
    
    Input: 
    series - a column to encode
    drop_first=False 
    Output: pd.DataFrame
    """
    encoded = OneHotEncoder(sparse_output=False).fit_transform(series.to_numpy().reshape((-1, 1)))
    unique_vals = [f'{series.name}_{i}' for i in sorted(series.unique())]
    encoded_df = pd.DataFrame(encoded, columns=unique_vals)
    if drop_first:
        encoded_df = encoded_df.drop(columns=encoded_df.columns[0])
    return encoded_df


def make_standard_scaling(series: pd.Series, mean=None, std=None, requires_statistics=False) -> pd.DataFrame:
    """
    Performs StandardScaling on a numerical Series
    """
    if mean is None and std is None:
        encoded = StandardScaler().fit_transform(series.to_numpy().reshape(-1, 1))
    else:
        X = series.to_numpy()
        encoded = (X - mean) / std
    
    encoded_df = pd.DataFrame(encoded, columns=[series.name])

    if requires_statistics:
        return encoded_df, series.mean(), series.std()
    else:
        return encoded_df


def titanic_make_initials_column(name_column: pd.Series) -> pd.Series:
    """
    Initials parsing from 'Name' column
    """
    initials = name_column.str.extract('([A-Za-z]+)\.').squeeze()

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

    column_prepared = initials.replace(unique_initials, initials_to_replace)

    return column_prepared


def titanic_fill_nulls(df: pd.DataFrame, is_train=True) -> pd.DataFrame:
    """
    Takes dataset and fills:
      - age nulls basing on mean age grouped by initials
      - fare nulls in test dataset
      - embarked nulls in train dataset
    """
    mean_age_by_initial = df.groupby('Initial')['Age'].mean()

    for initial in mean_age_by_initial.keys():
        df.loc[(df['Age'].isnull()) & (df['Initial'] == initial), 'Age'] = mean_age_by_initial[initial]

    if is_train:
        df['Embarked'] = df['Embarked'].fillna('S')
    else:
        df.loc[(df['Fare'].isnull()), 'Fare'] = df.groupby('Pclass')['Fare'].mean()[3]

    return df



# Training

def save_cv_metrics(y_true, y_pred) -> dict:
    """
    Returns a dictinory like: 
    {
        'Accuracy': ,
        'Precision': ,
        'Recall': ,
        'F1': 
    }
    """
    return {'Accuracy': accuracy_score(y_true, y_pred), 'Precision': precision_score(y_true, y_pred), 'Recal': recall_score(y_true, y_pred), 'F1': f1_score(y_true, y_pred)}


def aggregate_cv_metrics(metrics: dict):
    """
    
    """
    metrics_df = pd.DataFrame(metrics)
    metrics_avg = pd.DataFrame(metrics_df.to_numpy().mean(axis=0).reshape(1, -1), columns=metrics_df.columns)

    return metrics_avg