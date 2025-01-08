import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from loguru import logger

# Загрузка конфигурации
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Логирование
logger.add(config['general']['training_log_path'], rotation="1 MB", level=config['general']['log_level'])


# ==========================
# 🤖 МОДЕЛЬ ДЛЯ МЕТА-ОБУЧЕНИЯ
# ==========================
class MAML:
    def __init__(self, model, inner_lr, outer_lr, adaptation_steps):
        """
        Инициализация MAML.

        Args:
            model (nn.Module): Нейронная сеть для обучения.
            inner_lr (float): Внутренняя скорость обучения.
            outer_lr (float): Внешняя скорость обучения.
            adaptation_steps (int): Количество шагов адаптации.
        """
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.adaptation_steps = adaptation_steps
        self.outer_optimizer = optim.Adam(self.model.parameters(), lr=self.outer_lr)
    
    def adapt(self, task_data):
        """
        Адаптация модели на новой задаче.

        Args:
            task_data (tuple): Данные задачи (inputs, targets).

        Returns:
            nn.Module: Адаптированная модель.
        """
        inputs, targets = task_data
        model_copy = self._clone_model()
        inner_optimizer = optim.SGD(model_copy.parameters(), lr=self.inner_lr)
        
        for step in range(self.adaptation_steps):
            predictions = model_copy(inputs)
            loss = nn.CrossEntropyLoss()(predictions, targets)
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()
        
        return model_copy
    
    def meta_update(self, tasks):
        """
        Мета-обновление модели на основе нескольких задач.

        Args:
            tasks (list): Список задач для обучения.
        """
        meta_loss = 0.0
        self.outer_optimizer.zero_grad()
        
        for task_data in tasks:
            inputs, targets = task_data
            adapted_model = self.adapt(task_data)
            predictions = adapted_model(inputs)
            loss = nn.CrossEntropyLoss()(predictions, targets)
            meta_loss += loss
        
        meta_loss /= len(tasks)
        meta_loss.backward()
        self.outer_optimizer.step()
        
        logger.info(f"🔄 MAML — Мета-лосс: {meta_loss.item():.4f}")
    
    def _clone_model(self):
        """
        Создает копию модели с теми же параметрами.
        """
        clone = type(self.model)().to(next(self.model.parameters()).device)
        clone.load_state_dict(self.model.state_dict())
        return clone
