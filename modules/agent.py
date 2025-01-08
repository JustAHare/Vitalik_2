import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import yaml
from loguru import logger

# Загрузка конфигурации
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Логирование
logger.add(config['general']['training_log_path'], rotation="1 MB", level=config['general']['log_level'])


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
