# Pong AI using Deep Q-Learning (DQN)

This project implements a Pong AI agent trained using Deep Q-Learning (DQN) in Python with PyTorch. The agent learns to play Pong against a human player by training on a simplified, pixel-free environment. The project supports both training and human vs AI gameplay using `pygame`.

## Features
- **DQN agent** implemented with PyTorch
- **Self-play training environment** (no images, uses game state features)
- **Headless mode** for fast training
- **Human vs AI gameplay** using `pygame`
- **Model checkpointing** (automatic save/load)
- **Simple, modular code** (single file)
- **Reward shaping** for improved learning

## Requirements
- Python 3.7+
- PyTorch
- NumPy
- pygame

### Install dependencies
```bash
pip install torch numpy pygame
```

## How It Works
- The environment is a simplified Pong game (no graphics during training).
- The AI controls the **right paddle**; the human controls the **left paddle**.
- The AI is trained using a DQN with experience replay and a target network.
- **Rewards:**
  - `+1` if AI scores
  - `-1` if player scores
  - `+0.1` for AI hitting the ball
  - `+0.01` small reward for good AI positioning

## Usage

### Training the AI
If no saved model (`dqn_pong_model.pth`) is found, the script will train the agent from scratch:

```bash
python "Pong RL Game.py"
```

- Training runs for 120,000 episodes by default (see `PRETRAIN_EPISODES` in the code).
- Progress is logged every 500 episodes.
- The trained model is saved to `dqn_pong_model.pth`.

### Playing Against the AI
Once trained, the script will load the model and launch the game window for human vs AI play.

- **Controls:**
  - Up Arrow – Move paddle up
  - Down Arrow – Move paddle down
- The game runs at ~70 FPS.
- The AI paddle is on the right; you (the human) are on the left.

## Files Overview
- `Pong RL Game.py`  — Main training and gameplay script
- `dqn_pong_model.pth` — Saved PyTorch model (created after training)

## Saving & Loading
- The agent automatically saves to `dqn_pong_model.pth` after training.
- When restarting the script, it will load this file (if present) and start the game.

## Customization
- You can adjust training parameters (episodes, learning rate, etc.) at the top of `Pong RL Game.py`.
- To retrain from scratch, delete `dqn_pong_model.pth` and rerun the script.

## License

MIT License. Feel free to use, modify, and share. 