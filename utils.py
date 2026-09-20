import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, r2_score, mean_squared_error, root_mean_squared_error
from scipy.stats import mode

import torch
from torch.utils.data import DataLoader
from objects.datasets import TitanicDatasetPrepareNN, HousesDatasetPrepareNN, random_split
from config import config


# EDA
def check_nulls(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isnull().sum().sort_values(ascending=False)
    return missing[missing > 0]


def visualize_target_dependency(df, x, y, hue, palette='Dark2'):
    """
    plots a sns scatterplot to visualize the relationship between X, Y and two additional cols
    df = pd.DataFrame
    x = df col to be plotted on the X axis
    y = df col to be plotted on the Y axis
    hue = df col to be used as hue in sns.scatterplot()
    """
    plt.figure(figsize = (6, 4))
    fig = sns.scatterplot(data = df, x = x, y = y, hue = hue, legend="full", palette = palette)
    fig.spines['top'].set_visible(False)
    fig.spines['right'].set_visible(False)
    plt.title(f"Effect of {x} and {hue} on {y}", weight = "bold", fontsize = 12)
    plt.ylabel(y, fontsize = 11)
    plt.xlabel(x, fontsize = 11)
    plt.show()

    
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


def houses_fill_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
     
    """
    cat_cols = list(config.cat_features.houses)
    num_cols = list(config.num_features.houses)

    df[cat_cols] = df[cat_cols].fillna('None')
    df[num_cols] = df[num_cols].fillna(0)

    return df


def make_dataloaders(task, df_train: pd.DataFrame, df_test: pd.DataFrame):
    if task == 'titanic':
        df_tr, df_val = train_test_split(
            df_train,
            test_size=0.2,
            stratify=df_train[config.targets.titanic],
            random_state=config.general.random_state
        )

        data_train = TitanicDatasetPrepareNN(df_tr)
        data_val = TitanicDatasetPrepareNN(df_val)
        data_test = TitanicDatasetPrepareNN(df_test)

    elif task == 'houses':
        df_tr, df_val = train_test_split(
            df_train,
            test_size=0.2,
            random_state=config.general.random_state
        )

        data_train = HousesDatasetPrepareNN(df_tr)
        data_val = HousesDatasetPrepareNN(df_val)
        data_test = HousesDatasetPrepareNN(df_test)

    train_loader = DataLoader(data_train, config.general.batch_size, shuffle=True)
    val_loader = DataLoader(data_val, config.general.batch_size, shuffle=False)
    test_loader = DataLoader(data_test, config.general.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

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
    return {'accuracy': accuracy_score(y_true, y_pred), 'precision': precision_score(y_true, y_pred), 'recal': recall_score(y_true, y_pred), 'f1': f1_score(y_true, y_pred)}


def metrics_to_string(stage, loss, accuracy, precision, recal, f1):
    return f"{stage.upper()}: loss={loss:.4f}, accuracy={accuracy:.4f}, precision={precision:.4f}, recal={recal:.4f}, f1={f1:.4f}"


def regression_metrics_to_string(stage, loss, mse, mae, r2, rmsle):
    return f"{stage.upper()}: loss={loss:.4f}, mse={mse:.4f}, mae={mae:.4f}, r2={r2:.4f}, rmsle={rmsle:.4f}"


def save_cv_regression_metrics(y_true, y_pred) -> dict:
    # y_true_clipped = np.clip(np.expm1(y_true), a_min=0, a_max=None)
    # y_pred_clipped = np.clip(np.expm1(y_pred), a_min=0, a_max=None)
    # y_true = np.expm1(y_true)
    # y_pred = np.expm1(y_pred)

    return {'mse': mean_squared_error(y_true, y_pred), 'mae': mean_absolute_error(y_true, y_pred), 'r2': r2_score(y_true, y_pred), \
            'rmse': root_mean_squared_error(y_true, y_pred)}

     

def aggregate_cv_metrics(metrics: dict):
    """
    
    """
    metrics_df = pd.DataFrame(metrics)
    metrics_avg = pd.DataFrame(metrics_df.to_numpy().mean(axis=0).reshape(1, -1), columns=metrics_df.columns)

    return metrics_avg


def make_classification_prediction(X_test, models):

    y_preds = []
    for model in models:
        y_pred = model.predict(X_test)
        y_preds.append(y_pred)

    y_preds = np.array(y_preds).T
    final_predictions = mode(y_preds, axis=1).mode

    return final_predictions


def make_regression_prediction(X_test, models):

    y_preds = []
    if len(models) == 1:
        y_pred = torch.expm1(models[0](X_test))
        return y_pred
    else:
        for model in models:
            y_pred = np.expm1(model.predict(X_test))
            y_preds.append(y_pred)

    y_preds = np.array(y_preds)
    final_predictions = np.average(y_preds, axis=0)

    return final_predictions
     


def save_predictions(predictions, indexes, file_name, column_names):

    file_name += '.csv'

    df = pd.DataFrame({column_names[0]: indexes,
                       column_names[1]: predictions})
    
    if os.path.exists(file_name) and os.path.isfile(file_name):
        os.remove(file_name)

    df.to_csv(file_name, index=False)


class EarlyStopping:
	""" Класс отслеживания значимого изменения таргетированного атрибута модели """
	def __init__(self, mode='min', patience=10, threshold=1e-4, threshold_mode='rel'):
		self.mode = mode
		self.patience = patience
		self.threshold = threshold
		self.threshold_mode = threshold_mode
		self.count = 0
		self.best = None
		
	def __call__(self, tracked_parameter):
		current = float(tracked_parameter)
		if self.best is None:
			self.best = current
			return False
		
		if self.changed_better(current, self.best):
			self.best = current
			self.count = 0
		else:
			self.count += 1
		
		return self.count >= self.patience
		
	def changed_better(self, current, best):
		if self.mode == 'min' and self.threshold_mode == 'rel':
			return current < best - best * self.threshold
		
		if self.mode == 'min' and self.threshold_mode == 'abs':
			return current < best - self.threshold
		
		if self.mode == 'max' and self.threshold_mode == 'rel':
			return current < best + best * self.threshold
		
		else: # self.mode == 'max' and self.threshold_mode == 'abs'
			return current < best + self.threshold