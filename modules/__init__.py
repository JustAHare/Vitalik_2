# Инициализация пакета modules

from .agent import Agent, ActorCriticLSTM
from .environment import ColorEnvironment
from .maml import MAML
from .utils import (
    save_to_pickle,
    load_from_pickle,
    calculate_accuracy,
    calculate_loss,
    save_model,
    load_model,
    ensure_directory_exists
)

__all__ = [
    "Agent",
    "ActorCriticLSTM",
    "ColorEnvironment",
    "MAML",
    "save_to_pickle",
    "load_from_pickle",
    "calculate_accuracy",
    "calculate_loss",
    "save_model",
    "load_model",
    "ensure_directory_exists"
]
