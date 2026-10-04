# Common pipeline for Titanic and House prices datasets
## Getting started 
*How to create environment and start the main pipeline:*
```
conda env create -f environment.yml
conda activate titanic-housing
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
python main.py
```

Basically, the environment contains a GPU-version of PyTorch, but device may be selected in `config.py`

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