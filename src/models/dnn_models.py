import torch
from torch import nn
import numpy as np
import pandas as pd
from pathlib import Path

from configs.config import config


class SimpNN(nn.Module):
    def __init__(self, input, output, hidden_size):
        super().__init__()
        self.layer1 = nn.Linear(input, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output)

    def forward(self, X):
        x = self.layer1(X)
        x = self.relu(x)
        x = self.layer2(x)

        return x


class MoreLayersNN(nn.Module):
    def __init__(self, input, output, hidden_size1, hidden_size2):
        super().__init__()
        self.layer1 = nn.Linear(input, hidden_size1)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.layer3 = nn.Linear(hidden_size2, output)

    def forward(self, X):
        x = self.layer1(X)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)

        return x


class ImprovedNN(nn.Module):
    def __init__(self, input, output, hidden_size1, hidden_size2):
        super().__init__() 
        self.layer1 = nn.Linear(input, hidden_size1)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.layer3 = nn.Linear(hidden_size2, output)
        self.bnrm1 = nn.BatchNorm1d(hidden_size1)
        self.bnrm2 = nn.BatchNorm1d(hidden_size2)
        self.drop = nn.Dropout(0.5)

    def forward(self, X):
        x = self.layer1(X)
        x = self.bnrm1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.layer2(x)
        x = self.bnrm2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.layer3(x)

        return x


class EarlyStopping:
    """
    Отслеживает улучшение метрики, сохраняет полный чекпоинт на диск
    и останавливает обучение при отсутствии прогресса.
    """
    def __init__(self,
                 checkpoint_name: str = 'best_checkpoint.pt',
                 mode: str = 'min',
                 patience: int = 10,
                 threshold: float = 1e-4,
                 threshold_mode: str = 'rel',
                 save_optimizer: bool = True):
        assert mode in ['min', 'max'], "Параметр mode должен быть 'min' или 'max'"
        assert threshold_mode in ['rel', 'abs'], "Параметр threshold_mode должен быть 'rel' или 'abs'"

        self.checkpoint_path = Path('checkpoints/' + checkpoint_name)
        self.mode = mode
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.save_optimizer = save_optimizer

        self.count = 0
        self.best = None
        self.best_epoch = 0
        self.best_metrics = {}

        # Создаем директорию для чекпоинта, если ее нет
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def changed_better(self, current: float, best: float) -> bool:
        if self.mode == 'min':
            if self.threshold_mode == 'rel':
                return current < best - abs(best) * self.threshold
            return current < best - self.threshold
        else:  # mode == 'max'
            if self.threshold_mode == 'rel':
                return current > best + abs(best) * self.threshold
            return current > best + self.threshold

    def __call__(self,
                 tracked_parameter: float,
                 model: torch.nn.Module,
                 epoch: int,
                 metrics: dict,
                 optimizer: torch.optim.Optimizer = None) -> bool:
        current = float(tracked_parameter)

        if self.best is None or self.changed_better(current, self.best):
            self.best = current
            self.best_epoch = epoch
            self.count = 0
            self.best_metrics = metrics.copy()
            self.best_metrics['best_epoch'] = epoch

            # Сохранение чекпоинта на диск
            self._save_checkpoint(model, optimizer, epoch, metrics)
        else:
            self.count += 1

        return self.count >= self.patience

    def _save_checkpoint(self,
                         model: torch.nn.Module,
                         optimizer: torch.optim.Optimizer,
                         epoch: int,
                         metrics: dict):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_score': self.best,
            'metrics': metrics
        }
        if self.save_optimizer and optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(checkpoint, self.checkpoint_path)

    def restore_best_weights(self, model: torch.nn.Module, device: str = None):
        """Загружает лучшие веса из сохраненного файла обратно в модель."""
        if not self.checkpoint_path.exists():
            print(f"Файл {self.checkpoint_path} не найден.")
            return

        map_location = torch.device(device) if device else None
        checkpoint = torch.load(self.checkpoint_path, map_location=map_location, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])