import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 100
BALL_SIZE = 10
PADDLE_SPEED = 8
BALL_SPEED = 5  
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 64
MEMORY_SIZE = 10000
PRETRAIN_EPISODES = 120000
TARGET_UPDATE = 1000
REPLAY_START_SIZE = 1000
MODEL_PATH = "dqn_pong_model.pth"
class PongEnv:
    def __init__(self, headless=False):
        self.headless = headless
        if not self.headless:
            pygame.init()
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption("Pong Game")
        self.reset()   
    def reset(self):
        self.ball_x = WINDOW_WIDTH // 2
        self.ball_y = WINDOW_HEIGHT // 2
        self.ball_dx = BALL_SPEED * random.choice([-1, 1])
        self.ball_dy = BALL_SPEED * random.choice([-1, 1])
        center = WINDOW_HEIGHT // 2 - PADDLE_HEIGHT // 2
        self.player_y = np.clip(center + random.randint(-100, 100), 0, WINDOW_HEIGHT - PADDLE_HEIGHT)
        self.ai_y = WINDOW_HEIGHT // 2 - PADDLE_HEIGHT // 2
        return self._get_state()
    def step(self, action):
        if action == 1:
            self.ai_y -= PADDLE_SPEED
        elif action == 2:
            self.ai_y += PADDLE_SPEED
        self.ai_y = np.clip(self.ai_y, 0, WINDOW_HEIGHT - PADDLE_HEIGHT)
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy
        if self.ball_y <= 0 or self.ball_y >= WINDOW_HEIGHT - BALL_SIZE:
            self.ball_dy *= -1
        reward = 0
        done = False
        if self.ball_x <= PADDLE_WIDTH:
            if self.player_y <= self.ball_y <= self.player_y + PADDLE_HEIGHT:
                self.ball_dx *= -1
            else:
                reward = 1
                done = True
        if self.ball_x >= WINDOW_WIDTH - PADDLE_WIDTH - BALL_SIZE:
            if self.ai_y <= self.ball_y <= self.ai_y + PADDLE_HEIGHT:
                self.ball_dx *= -1
                reward = 0.1
            else:
                reward = -1
                done = True
        if self.ball_dx > 0:
            if abs((self.ball_y + BALL_SIZE / 2) - (self.ai_y + PADDLE_HEIGHT / 2)) < 20:
                reward += 0.01
        reward = np.clip(reward, -1, 1)
        return self._get_state(), reward, done
    def _get_state(self):
        return np.array([
            self.ball_x / WINDOW_WIDTH,
            self.ball_y / WINDOW_HEIGHT,
            self.ball_dx / BALL_SPEED,
            self.ball_dy / BALL_SPEED,
            self.player_y / WINDOW_HEIGHT,
            self.ai_y / WINDOW_HEIGHT
        ], dtype=np.float32)
    def render(self):
        if self.headless:
            return
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (255, 255, 255), (0, self.player_y, PADDLE_WIDTH, PADDLE_HEIGHT))
        pygame.draw.rect(self.screen, (255, 255, 255), (WINDOW_WIDTH - PADDLE_WIDTH, self.ai_y, PADDLE_WIDTH, PADDLE_HEIGHT))
        pygame.draw.rect(self.screen, (255, 255, 255), (self.ball_x, self.ball_y, BALL_SIZE, BALL_SIZE))
        pygame.display.flip()
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
    def forward(self, x):
        return self.fc(x)
class Agent:
    def __init__(self):
        self.model = DQN()
        self.target_model = DQN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.steps = 0
        self.epsilon = 1.0
    def remember(self, s, a, r, s_, d):
        self.memory.append((s, a, r, s_, d))
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            return torch.argmax(self.model(state)).item()
    def train_step(self):
        if len(self.memory) < max(BATCH_SIZE, REPLAY_START_SIZE):
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q = self.target_model(next_states).max(1)[0]
        target_q = rewards + GAMMA * next_q * (~dones)
        loss = nn.functional.mse_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.steps % TARGET_UPDATE == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        self.steps += 1
        self.epsilon = max(0.05, self.epsilon * 0.999)
    def save(self, path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'is_trained': True
        }
        torch.save(checkpoint, path)
    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'target_model_state_dict' in checkpoint:
                self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            else:
                self.target_model.load_state_dict(checkpoint['model_state_dict'])            
            self.steps = checkpoint.get('steps', 0)
            self.is_trained = checkpoint.get('is_trained', True)
        else:
            self.model.load_state_dict(checkpoint)
            self.target_model.load_state_dict(checkpoint)
            self.is_trained = True        
        self.model.eval()
        self.target_model.eval()
if __name__ == "__main__":
    env = PongEnv(headless=True)
    agent = Agent()
    if not os.path.exists(MODEL_PATH):
        print("Training DQN from scratch...")
        for episode in range(PRETRAIN_EPISODES):
            state = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.train_step()
                state = next_state
                total_reward += reward
            if episode % 500 == 0:
                print(f"Episode {episode}: Total Reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.2f}")
        agent.save(MODEL_PATH)
        print("Training completed and model saved.")
    else: 
        print("Loading model...")
        agent.load(MODEL_PATH)
        agent.epsilon = 0.0 
        print("Model loaded.")
        with torch.no_grad():
            dummy_state = np.zeros(6, dtype=np.float32)
            agent.act(dummy_state)
    print("Starting User vs AI Gameplay...")
    env = PongEnv(headless=False)
    clock = pygame.time.Clock()
    state = env.reset()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            env.player_y -= PADDLE_SPEED
        if keys[pygame.K_DOWN]:
            env.player_y += PADDLE_SPEED
        env.player_y = np.clip(env.player_y, 0, WINDOW_HEIGHT - PADDLE_HEIGHT)
        action = agent.act(state)
        state, _, done = env.step(action)
        env.render()
        if done:
            state = env.reset()
        clock.tick(70)
