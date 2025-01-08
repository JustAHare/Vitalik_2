import os
import pickle
import yaml
from loguru import logger
import torch

# Загрузка конфигурации
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Логирование
logger.add(config['general']['training_log_path'], rotation="1 MB", level=config['general']['log_level'])


# ==========================
# 📝 ФУНКЦИИ ДЛЯ РАБОТЫ С ФАЙЛАМИ
# ==========================
def save_to_pickle(data, path):
    """
    Сохраняет данные в pickle-файл.
    """
    try:
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"💾 Данные успешно сохранены в {path}")
    except Exception as e:
        logger.error(f"❌ Ошибка при сохранении данных: {e}")


def load_from_pickle(path):
    """
    Загружает данные из pickle-файла.
    """
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"📥 Данные успешно загружены из {path}")
        return data
    except FileNotFoundError:
        logger.warning(f"⚠️ Файл {path} не найден. Возвращён пустой список.")
        return []
    except Exception as e:
        logger.error(f"❌ Ошибка при загрузке данных: {e}")
        return []


# ==========================
# 📊 ФУНКЦИИ ДЛЯ МЕТРИК
# ==========================
def calculate_accuracy(predictions, targets):
    """
    Рассчитывает точность предсказаний.
    """
    correct = (predictions == targets).sum().item()
    accuracy = correct / len(targets) if len(targets) > 0 else 0.0
    logger.info(f"📊 Точность: {accuracy:.2f}")
    return accuracy


def calculate_loss(loss_fn, predictions, targets):
    """
    Рассчитывает потери на основе функции потерь.
    """
    loss = loss_fn(predictions, targets)
    logger.info(f"📉 Потери: {loss.item():.4f}")
    return loss.item()


# ==========================
# 💾 ФУНКЦИИ ДЛЯ РАБОТЫ С МОДЕЛЯМИ
# ==========================
def save_model(model, path):
    """
    Сохраняет модель PyTorch.
    """
    try:
        torch.save(model.state_dict(), path)
        logger.info(f"💾 Модель успешно сохранена в {path}")
    except Exception as e:
        logger.error(f"❌ Ошибка при сохранении модели: {e}")


def load_model(model, path, device='cpu'):
    """
    Загружает модель PyTorch.
    """
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        logger.info(f"📥 Модель успешно загружена из {path}")
    except FileNotFoundError:
        logger.warning(f"⚠️ Файл модели {path} не найден.")
    except Exception as e:
        logger.error(f"❌ Ошибка при загрузке модели: {e}")


# ==========================
# 📂 УТИЛИТЫ ДЛЯ ДИРЕКТОРИЙ
# ==========================
def ensure_directory_exists(path):
    """
    Создает директорию, если она не существует.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"📂 Директория {path} была создана.")


# ==========================
# 🚀 ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ==========================
if __name__ == '__main__':
    # Пример сохранения данных
    test_data = {'цвета': ['зеленый', 'синий', 'золотой']}
    save_to_pickle(test_data, './data/test_data.pkl')
    
    # Пример загрузки данных
    loaded_data = load_from_pickle('./data/test_data.pkl')
    print("Загруженные данные:", loaded_data)
    
    # Пример расчёта метрик
    import torch.nn.functional as F
    predictions = torch.tensor([1, 0, 1])
    targets = torch.tensor([1, 0, 0])
    accuracy = calculate_accuracy(predictions, targets)
    loss = calculate_loss(F.mse_loss, predictions.float(), targets.float())
    
    # Пример сохранения и загрузки модели
    import torch.nn as nn
    test_model = nn.Linear(3, 3)
    save_model(test_model, './models/test_model.pth')
    load_model(test_model, './models/test_model.pth')
    
    # Пример создания директории
    ensure_directory_exists('./logs/test_logs')
