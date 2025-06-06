# Pong Game using Deep Q-Learning

This project implements a reinforcement learning-based Pong AI agent trained using Deep Q-Learning (DQN) in Python with PyTorch. There's both training and human vs AI gameplay implemented here, using `pygame`.

## Features
- DQN agent implemented with PyTorch
- Self-play AI training environment
- Human vs AI gameplay using `pygame`
- Reward shaping for improved and efficient learning

## Requirements
- Python 3.7+
- PyTorch
- NumPy
- pygame

### Install dependencies
```bash
pip install torch numpy pygame
```

## Usage

### Training the AI
If no saved model (`dqn_pong_model.pth`) is found, the script will train the agent from scratch.

- Training runs for 120,000 episodes by default (refer to `PRETRAIN_EPISODES` in the code).
- Progress is logged every 500 episodes.
- The trained model is saved to `dqn_pong_model.pth`.

### Playing Against the AI
Once trained, the script will load the model and launch the game window for human vs AI gameplay.

- **Controls:**
  - Up Arrow – Move paddle up
  - Down Arrow – Move paddle down
- The game runs at ~70 FPS.
- The AI paddle is on the right; and the human (user) is on the left.

## Files Overview
- `Pong DQN Game.py`  — Main training and gameplay script
- `dqn_pong_model.pth` — Saved PyTorch model (created after training)

## Saving & Loading
- The agent automatically saves to `dqn_pong_model.pth` after training.
- When restarting the script, it will load this file (if present in the same directory) and start the game.

## Customization
- You may adjust the training parameters (episodes, FPS, learning rate, etc.) in the `Pong DQN Game.py` python script.
- To retrain from scratch, delete `dqn_pong_model.pth` and rerun the script.

## Contributing

Contributions are welcome!

## License

Distributed under the MIT License.  
