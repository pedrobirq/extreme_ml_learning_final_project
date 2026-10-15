# Common pipeline for Titanic and House prices datasets
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


## Exploring the pipeline

- `main.py` - the main script that starts the whole pipeline of training models for both tasks and shows a table of metrics;
- `config.py` - configuration of the pipeline and models;
- `utils.py` - a set of fuctions and classes used through the whole pipeline;
- `train_functions.py` - a set of functions to train models;
- `objects/datasets.py` - dataset preparation code;
- `objects/dnn_models.py` - deep learning model architectures;
- `EDA.ipynb` - exploration Data Analysis for both tasks; 
- `draft.ipynb` - just some random code.

## Competition results
- The best metric for the Titanic competition is `0.77990` obtained from simple neural network with two layers.
- The best metric for the House Prices competition is `0.12044` obtained from CatBoost. 

