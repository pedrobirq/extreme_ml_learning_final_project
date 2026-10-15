# The titanic competition pipeline

## Getting started 
*How to create environment and start the main pipeline:*
```
uv sync
uv run main.py
```

Basically, the environment contains a GPU-version of PyTorch, but device may be selected in `config.py`

## How to use this pipeline

### Data preprocessing

- `src/datasets/titanic.py` contains the preparation class `TitanicDatasetPrepare` which perfoms data preprocessing based on columnns with their preprocessing procedure that is demanded to coduct. `TitanicDatasetPrepareNN` is a wrapper function that takes fully preprocessed dataframe for using it in dataloaders further.
- `make_folds` function in `src/train_functions.py` performs preprocessing with the transmitted parameters regarded to classic models or nn models. It gives as an output a dictionary of statistics for each folder and:
    - `[fold_X_trains, fold_X_vals, fold_y_trains, fold_y_vals]` for classic models;
    - `[fold_train_loaders, fold_val_loaders]` for nn models.

### Training

- The function `train_classification` takes a model, it parameters, folders and nn_attributes (if a given model is a nn). If it's a nn, the function calls the `train_nn_classification` function. As an output it gives cross validation metrics and trained models for each folder.

### Test predicting

- The `make_test_predictions` function conducts test dataset preparation with given statistics and preprocessing configuration, evaluates each folder model and aggregates final prediction saving it to a `.csv` file.   

### Optional

#### Optuna

- Optuna may be used in this pipeline to find proper parameters for classic models. To use it, run this in the `main.py` after folders being prepared:

```python
run_optuna(model_name=MODEL_NAME,
            target_metric='val_f1',
            folds=folds)
```

## Competition results

| №   | Approach                                                                          | CV       | CV STD   | LB      |
| --- | --------------------------------------------------------------------------------- | -------- | -------- | ------- |
| 1   | LogisticRegression                                                                | 0.767934 | 0.035774 | 0.78229 |
| 2   | SimpNN                                                                            | 0.761926 | 0.014176 | 0.78708 |
| 3   | ImprovedNN                                                                        | 0.752124 | 0.024689 | 0.78229 |
| 4   | ImprovedNN + CosineLR                                                             | 0.758660 | 0.033222 | 0.77033 |
| 5   | LogReg + Age and Fare features distrubition normalized with `QuantileTransformer` | 0.768418 | 0.020882 | 0.77990 |
| 6   | LogReg + only Fare feature distrubition normalized with `QuantileTransformer`     | 0.770917 | 0.023894 | 0.77751 |
| 7   | CatBoostClassifier optuna tuned                                                   | 0.765641 | 0.038901 | 0.75358 |
| **8**   | **LogisticRegression optuna tuned with elasticnet**                                   | 0.771410 | 0.018839 | 0.78708 |


