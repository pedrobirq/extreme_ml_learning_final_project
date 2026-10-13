import torch
from torch import nn
import numpy as np
import pandas as pd
from pathlib import Path
import copy
from typing import Dict, Optional, Union


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
    Отслеживает метрику, кэширует лучшие веса в RAM (CPU) или на диск
    и сигнализирует об остановке при отсутствии прогресса.
    """
    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        mode: str = "min",
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        save_to_disk: bool = True,
        save_optimizer: bool = False,
    ):
        assert mode in ["min", "max"], "Параметр mode должен быть 'min' или 'max'"
        assert threshold_mode in ["rel", "abs"], "Параметр threshold_mode должен быть 'rel' или 'abs'"

        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.mode = mode
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.save_to_disk = save_to_disk
        self.save_optimizer = save_optimizer

        # Инициализация внутреннего состояния
        self.reset(self.checkpoint_path)

    def reset(self, checkpoint_path: Optional[Union[str, Path]] = None):
        """Сбрасывает состояние для нового фолда или нового испытания."""
        if checkpoint_path is not None:
            self.checkpoint_path = Path(checkpoint_path)

        if self.save_to_disk and self.checkpoint_path is not None:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        self.count = 0
        self.best: Optional[float] = None
        self.best_epoch = 0
        self.best_metrics: Dict[str, float] = {}
        self.best_state_dict: Optional[dict] = None
        self.best_optimizer_state_dict: Optional[dict] = None
        self.early_stop = False

    def changed_better(self, current: float, best: float) -> bool:
        if self.mode == "min":
            if self.threshold_mode == "rel":
                return current < best - abs(best) * self.threshold
            return current < best - self.threshold
        else:  # mode == 'max'
            if self.threshold_mode == "rel":
                return current > best + abs(best) * self.threshold
            return current > best + self.threshold

    def __call__(
        self,
        tracked_parameter: float,
        model: torch.nn.Module,
        epoch: int,
        metrics: dict,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> bool:
        current = float(tracked_parameter)

        if self.best is None or self.changed_better(current, self.best):
            self.best = current
            self.best_epoch = epoch
            self.count = 0
            self.best_metrics = metrics.copy()
            self.best_metrics["best_epoch"] = epoch

            # 1. Быстрое сохранение весов в оперативной памяти (CPU RAM)
            # clone() и detach() разрывают вычислительный граф и изолируют тензоры
            self.best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            if self.save_optimizer and optimizer is not None:
                self.best_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())

            # 2. Опциональная запись на диск (только если явно требуется)
            if self.save_to_disk and self.checkpoint_path is not None:
                self._save_checkpoint(epoch, metrics)
        else:
            self.count += 1
            if self.count >= self.patience:
                self.early_stop = True

        return self.early_stop

    def _save_checkpoint(self, epoch: int, metrics: dict):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.best_state_dict,
            "best_score": self.best,
            "metrics": metrics,
        }
        if self.save_optimizer and self.best_optimizer_state_dict is not None:
            checkpoint["optimizer_state_dict"] = self.best_optimizer_state_dict

        torch.save(checkpoint, self.checkpoint_path)

    def restore_best_weights(
        self, model: torch.nn.Module, device: Optional[Union[str, torch.device]] = None
    ):
        """
        Восстанавливает лучшие веса.
        В приоритете использует RAM, при отсутствии — считывает файл с диска.
        """
        target_device = device if device is not None else next(model.parameters()).device

        # Восстановление из RAM
        if self.best_state_dict is not None:
            model.load_state_dict(
                {k: v.to(target_device) for k, v in self.best_state_dict.items()}
            )
            return

        # Запасной вариант: восстановление с диска
        if self.save_to_disk and self.checkpoint_path and self.checkpoint_path.exists():
            checkpoint = torch.load(
                self.checkpoint_path, map_location=target_device, weights_only=True
            )
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            print("[EarlyStopping] Предупреждение: лучшие веса не найдены ни в памяти, ни на диске.")