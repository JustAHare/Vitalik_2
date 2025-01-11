import yaml
import os
import pickle
import logging

# Настройка стандартного логгера
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Логирование заменено на стандартное.")

# Загрузка конфигурации
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Пути к данным
TRAINING_DATA_PATH = config['paths']['training_data']

# Проверка и создание необходимых файлов
if not os.path.exists(os.path.dirname(TRAINING_DATA_PATH)):
    os.makedirs(os.path.dirname(TRAINING_DATA_PATH))

if not os.path.exists(TRAINING_DATA_PATH):
    with open(TRAINING_DATA_PATH, 'wb') as f:
        pickle.dump([], f)
        logger.info("Файл training_data.pkl создан.")

def load_training_data():
    """Загрузка обучающих данных"""
    try:
        with open(TRAINING_DATA_PATH, 'rb') as f:
            data = pickle.load(f)
            logger.info(f"Загружено {len(data)} записей из training_data.pkl.")
            return data
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {e}")
        return []

def save_training_data(data):
    """Сохранение обучающих данных"""
    try:
        with open(TRAINING_DATA_PATH, 'wb') as f:
            pickle.dump(data, f)
            logger.info("Данные успешно сохранены.")
    except Exception as e:
        logger.error(f"Ошибка при сохранении данных: {e}")
        print(f"❌ Ошибка при сохранении данных: {e}")

def manual_input_loop():
    """
    Основной цикл ручного ввода данных.
    """
    training_data = load_training_data()
    MAX_DATA_SIZE = 100000  # Максимальное количество записей
    print("\n🔵🟢🟡 Режим ручного ввода данных активирован!")
    print("Введите последовательность цветов вручную (зеленый, синий, золотой).")
    print("Для выхода введите 'exit'. Для просмотра данных введите 'show'.\n")
    
    while True:
        user_input = input("Введите цвет (зеленый/синий/золотой): ").strip().lower()
        
        if not user_input:
            print("⚠️ Пустой ввод. Попробуйте снова.")
            continue
        
        if user_input == 'exit':
            print("🚪 Выход из режима ручного ввода.")
            break
        elif user_input == 'show':
            print("📊 Текущие данные для обучения:")
            for i, color in enumerate(training_data[-10:], 1):
                print(f"{i}. {color}")
            continue
        elif user_input not in ['зеленый', 'синий', 'золотой']:
            print("❌ Некорректный ввод! Введите 'зеленый', 'синий' или 'золотой'.")
            continue
        
        # Лимит на количество данных
        if len(training_data) >= MAX_DATA_SIZE:
            print(f"⚠️ Набор данных достиг лимита в {MAX_DATA_SIZE} записей. Новые записи не принимаются.")
            continue
        
        # Добавляем данные в обучающий набор
        training_data.append(user_input)
        save_training_data(training_data)
        
        logger.info(f"Добавлен цвет: {user_input}")
        print(f"✅ Цвет '{user_input}' добавлен в набор данных.")

if __name__ == '__main__':
    try:
        manual_input_loop()
    except KeyboardInterrupt:
        print("\n🚨 Принудительное завершение программы.")
    except Exception as e:
        logger.error(f"Произошла ошибка: {e}")
        print(f"❌ Ошибка: {e}")
