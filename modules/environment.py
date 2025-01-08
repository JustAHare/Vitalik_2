import random
import yaml
from loguru import logger

# Загрузка конфигурации
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Логирование
logger.add(config['general']['training_log_path'], rotation="1 MB", level=config['general']['log_level'])

class ColorEnvironment:
    """
    Виртуальная среда для агента, взаимодействующего с последовательностями цветов.
    """
    def __init__(self):
        self.sequence = []  # История последовательностей
        self.current_step = 0
        self.colors = ['зеленый', 'синий', 'золотой']
        self.reset()
    
    def reset(self):
        """
        Сброс среды в начальное состояние.
        """
        self.sequence = [random.choice(self.colors) for _ in range(3)]
        self.current_step = 0
        logger.info("🌟 Среда была сброшена.")
        return self._get_state()
    
    def _get_state(self):
        """
        Возвращает текущее состояние среды.
        """
        return self.sequence[-3:]
    
    def step(self, action):
        """
        Выполняет действие агента.
        
        Args:
            action (str): Действие агента ('зеленый', 'синий', 'золотой').
        
        Returns:
            tuple: (состояние, награда, завершено)
        """
        correct_color = random.choice(self.colors)  # Генерация "правильного" цвета
        self.sequence.append(correct_color)
        self.current_step += 1
        
        if action == correct_color:
            reward = 1  # Агент угадал цвет
            logger.info(f"✅ Агент угадал цвет: {correct_color}")
        else:
            reward = -1  # Агент ошибся
            logger.warning(f"❌ Агент ошибся. Ожидалось: {correct_color}, получено: {action}")
        
        done = self.current_step >= config['training']['epochs']
        next_state = self._get_state()
        
        return next_state, reward, done
    
    def render(self):
        """
        Визуализирует текущее состояние среды.
        """
        print(f"🟢 Текущая последовательность: {self.sequence[-3:]}")
