import os
import subprocess
import yaml
from loguru import logger

# Загрузка конфигурации
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Логирование
logger.add(config['general']['training_log_path'], rotation="1 MB", level=config['general']['log_level'])

# Пути к файлам и скриптам
MANUAL_INPUT_SCRIPT = "manual_input.py"
AGENT_SCRIPT = "agent.py"
ENVIRONMENT_SCRIPT = "environment.py"
MAML_SCRIPT = "maml.py"
DATA_PATH = config['paths']['training_data']
MODEL_PATH = config['general']['save_model_path']
META_MODEL_PATH = config['general']['save_meta_model_path']
LOG_PATH = config['general']['training_log_path']

# ==========================
# 🛠️ УТИЛИТЫ
# ==========================
def clear_data():
    """
    Сбрасывает данные и модели.
    """
    if os.path.exists(DATA_PATH):
        os.remove(DATA_PATH)
        logger.info("🗑️ Данные обучения удалены.")
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
        logger.info("🗑️ Модель удалена.")
    if os.path.exists(META_MODEL_PATH):
        os.remove(META_MODEL_PATH)
        logger.info("🗑️ Мета-модель удалена.")
    print("✅ Данные и модели были сброшены.")


def view_logs():
    """
    Выводит последние 20 строк логов.
    """
    if os.path.exists(LOG_PATH):
        print("\n📜 Последние записи из логов:")
        with open(LOG_PATH, 'r') as log_file:
            lines = log_file.readlines()
            for line in lines[-20:]:
                print(line.strip())
    else:
        print("⚠️ Файл логов не найден.")


# ==========================
# 📊 ОСНОВНОЕ МЕНЮ
# ==========================
def main_menu():
    while True:
        print("\n🔵🟢🟡 Добро пожаловать в Виталика! 🟡🟢🔵")
        print("1. Ручной ввод данных")
        print("2. Обучение агента")
        print("3. Предсказание")
        print("4. Сброс данных и моделей")
        print("5. Просмотр логов")
        print("6. Выход")
        
        choice = input("\nВведите номер действия: ").strip()
        
        if choice == '1':
            print("\n🚀 Запуск ручного ввода данных...")
            subprocess.run(["python", MANUAL_INPUT_SCRIPT])
        elif choice == '2':
            print("\n🚀 Запуск обучения агента...")
            subprocess.run(["python", AGENT_SCRIPT])
        elif choice == '3':
            print("\n🚀 Запуск предсказания...")
            subprocess.run(["python", ENVIRONMENT_SCRIPT])
        elif choice == '4':
            print("\n🗑️ Сброс данных и моделей...")
            clear_data()
        elif choice == '5':
            print("\n📜 Просмотр логов...")
            view_logs()
        elif choice == '6':
            print("\n👋 До свидания!")
            break
        else:
            print("❌ Некорректный ввод. Пожалуйста, введите число от 1 до 6.")


if __name__ == '__main__':
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n🚨 Программа прервана пользователем.")
    except Exception as e:
        logger.error(f"❌ Произошла ошибка в лаунчере: {e}")
        print(f"❌ Ошибка: {e}")
