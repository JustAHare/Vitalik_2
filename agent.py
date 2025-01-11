import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import yaml
import logging
import pickle
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Загрузка конфигурации
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# ==========================
# 🤖 АРХИТЕКТУРА АКТОР-КРИТИК
# ==========================
class ActorCriticLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(ActorCriticLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.actor = nn.Linear(hidden_dim, output_dim)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # Проверяем и изменяем форму входного тензора
        if len(x.shape) == 2:  # Если (batch_size, input_size)
            x = x.unsqueeze(1)  # Добавляем seq_length = 1

        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Берём последнее скрытое состояние
        policy = F.softmax(self.actor(lstm_out), dim=-1)
        value = self.critic(lstm_out)
        return policy, value

# ==========================
# 🛠️ АГЕНТ
# ==========================
class Agent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = config['model']['hidden_size']
        self.num_layers = config['model']['num_layers']
        self.gamma = config['training']['gamma']
        self.lr = config['training']['learning_rate']
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCriticLSTM(self.state_dim, self.hidden_dim, self.action_dim, self.num_layers).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        policy, _ = self.model(state)
        action = torch.multinomial(policy, 1).item()
        return action
    
    def update(self, rewards, log_probs, values, dones):
        returns = []
        discounted_sum = 0
        
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)
        
        returns = torch.FloatTensor(returns).to(self.device)
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)
        
        advantage = returns - values.squeeze()
        actor_loss = -(log_probs * advantage).mean()
        critic_loss = F.mse_loss(values.squeeze(), returns)
        loss = actor_loss + critic_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), config['training']['clip_grad_norm'])
        self.optimizer.step()
        logger.info(f"🔄 Актор-лосс: {actor_loss.item():.4f}, Критик-лосс: {critic_loss.item():.4f}")
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        logger.info(f"💾 Модель сохранена по пути: {path}")
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"📥 Модель загружена из файла: {path}")

# ==========================
# 🚀 ОБУЧЕНИЕ АГЕНТА
# ==========================
COLOR_MAP = {'зеленый': 0, 'синий': 1, 'золотой': 2}

def encode_sequence(sequence):
    """Кодирует строковую последовательность цветов в числовой формат."""
    return [COLOR_MAP[color] for color in sequence if color in COLOR_MAP]

def evaluate_model(agent, training_data):
    """
    Оценивает текущую модель на тренировочных данных.
    Возвращает среднюю награду.
    """
    if not training_data:
        logger.warning("Данные для оценки отсутствуют. Метрика не вычислена.")
        return float('-inf')

    training_data_encoded = encode_sequence(training_data)
    total_reward = 0

    for color in training_data_encoded:
        state = torch.FloatTensor([[color]]).to(agent.device)  # seq_length = 1
        action = agent.select_action([color])
        reward = 1.0 if action == color else -1.0
        total_reward += reward

    avg_reward = total_reward / len(training_data_encoded)
    logger.info(f"Оценка модели завершена. Средняя награда: {avg_reward:.4f}")
    return avg_reward

def train_agent(agent, training_data):
    logger.info("Начало обучения агента.")

    # Загрузка сохранённой модели, если она существует
    model_path = config['general']['save_model_path']
    best_metric = float('-inf')  # Инициализируем худшей возможной метрикой

    if os.path.exists(model_path):
        agent.load_model(model_path)
        logger.info(f"📥 Загружена предыдущая лучшая модель из {model_path}.")
        best_metric = evaluate_model(agent, training_data)  # Оценка сохранённой модели
        logger.info(f"🔄 Метрика загруженной модели: {best_metric:.4f}")
    else:
        logger.info("📥 Сохранённая модель не найдена. Начинаем обучение с нуля.")

    if not training_data:
        logger.warning("Данные для обучения отсутствуют. Добавьте данные и повторите попытку.")
        return

    # Кодирование данных
    logger.info(f"Обработка данных: {len(training_data)} записей.")
    training_data_encoded = encode_sequence(training_data)

    for epoch in range(config['training']['epochs']):
        logger.info(f"Эпоха {epoch + 1}/{config['training']['epochs']} началась.")
        log_probs, values, rewards, dones = [], [], [], []

        epoch_rewards = 0  # Накопление награды за эпоху

        for color in training_data_encoded:
            # Обеспечение корректной формы данных
            state = torch.FloatTensor([[color]]).to(agent.device)  # seq_length = 1
            action = agent.select_action([color])
            reward = 1.0 if action == color else -1.0
            done = False

            epoch_rewards += reward  # Накопление награды

            policy, value = agent.model(state)
            log_prob = torch.log(policy.squeeze(0)[action])

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            dones.append(done)

        # Обновление агента
        agent.update(rewards, log_probs, values, dones)

        # Средняя награда за эпоху
        avg_reward = epoch_rewards / len(training_data_encoded)
        logger.info(f"Средняя награда за эпоху {epoch + 1}: {avg_reward:.4f}")

        # Сохранение лучшей модели
        if avg_reward > best_metric:
            best_metric = avg_reward
            agent.save_model(model_path)
            logger.info(f"💾 Лучшая модель сохранена с метрикой: {best_metric:.4f}")

        if (epoch + 1) % config['metrics']['log_interval'] == 0:
            logger.info(f"Эпоха {epoch + 1} завершена.")

    logger.info("Обучение завершено.")


if __name__ == '__main__':
    agent = Agent(state_dim=1, action_dim=3)
    logger.info("Агент успешно инициализирован.")
    
    with open(config['paths']['training_data'], 'rb') as f:
        training_data = pickle.load(f)
    train_agent(agent, training_data)
