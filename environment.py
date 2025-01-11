import os
import torch
import yaml
import logging
import pickle
from agent import Agent

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Загрузка конфигурации
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Пути к данным и модели
MODEL_PATH = config['general']['save_model_path']
TRAINING_DATA_PATH = config['paths']['training_data']

COLOR_MAP = {'зеленый': 0, 'синий': 1, 'золотой': 2}
REVERSE_COLOR_MAP = {v: k for k, v in COLOR_MAP.items()}

def predict_color():
    """
    Загружает модель и делает предсказания.
    """
    if not torch.cuda.is_available():
        logger.info("⚠️ CUDA не доступна. Используется CPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Проверка наличия модели
    if not os.path.exists(MODEL_PATH):
        logger.error("Модель не найдена. Сначала выполните обучение.")
        print("❌ Ошибка: модель не найдена. Сначала выполните обучение.")
        return

    # Загрузка модели
    logger.info("Загрузка модели...")
    agent = Agent(state_dim=1, action_dim=3)
    agent.load_model(MODEL_PATH)

    print("\n🔵🟢🟡 Режим предсказания активирован!")
    print("Введите последовательность цветов для предсказания. Для выхода введите 'exit'.\n")

    while True:
        user_input = input("Введите последовательность цветов (через пробел): ").strip().lower()

        if user_input == 'exit':
            print("🚪 Выход из режима предсказания.")
            break

        try:
            sequence = user_input.split()
            encoded_sequence = [COLOR_MAP[color] for color in sequence if color in COLOR_MAP]

            if not encoded_sequence:
                print("⚠️ Некорректный ввод. Введите последовательность из цветов: 'зеленый', 'синий', 'золотой'.")
                continue

            # Предсказание следующего цвета
            state = torch.FloatTensor(encoded_sequence[-1:]).unsqueeze(0).to(device)
            action = agent.select_action(state.tolist())
            predicted_color = REVERSE_COLOR_MAP[action]

            print(f"🔮 Предсказанный цвет: {predicted_color}")
        except Exception as e:
            logger.error(f"Ошибка во время предсказания: {e}")
            print(f"❌ Ошибка: {e}")

if __name__ == '__main__':
    try:
        predict_color()
    except KeyboardInterrupt:
        print("\n🚨 Принудительное завершение программы.")
    except Exception as e:
        logger.error(f"Произошла ошибка: {e}")
        print(f"❌ Ошибка: {e}")
