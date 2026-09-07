# Common pipeline for Titanic and House prices datasets
## Getting started 
Окружение и зависимости подтягиваются через:
```
conda env create -f environment.yml
conda activate titanic-housing
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements-torch.txt   # если версии зафиксированы отдельно
```
