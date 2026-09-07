import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


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
        return encoded_df, encoded.mean(), encoded.std()
    else:
        return encoded_df