# RL Assignment — Text Flappy Bird Agents

> **Course:** Reinforcement Learning  
> **Author:** Daniel Fernandez de la Mela 
> **Deadline:** April 10, 2026

## 📋 Project Description

Implementation and comparison of two reinforcement learning agents on the [Text Flappy Bird](https://gitlab-research.centralesupelec.fr/stergios.christodoulidis/text-flappy-bird-gym) (TFB) environment:

1. **Agent 1 — Constant-α First-Visit MC Control** with ε-greedy exploration
2. **Agent 2 — True Online Sarsa(λ)** with tile coding (Sutton & Barto §12.7)

Both agents are evaluated on `TextFlappyBird-v0` (distance observations) and `TextFlappyBird-screen-v0` (screen observations). The project includes parameter sweeps, generalization experiments, and value function visualizations.

## 🏗️ Project Structure

```
RL-Assignment/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── notebook.ipynb              # Main deliverable notebook
├── src/
│   ├── agents/
│   │   ├── mc_agent.py         # Monte Carlo Control agent
│   │   └── sarsa_lambda.py     # True Online Sarsa(λ) agent
│   └── utils/
│       ├── tile_coding.py      # Tile coding for function approximation
│       ├── plotting.py         # Visualization utilities
│       └── experiment.py       # Training loops & parameter sweeps
├── results/
│   ├── figures/                # Generated plots
│   └── models/                 # Saved weights / Q-tables
├── report/                     # LNCS LaTeX report
└── assets/                     # Demo GIFs
```

## 🚀 Installation

```bash
# Clone and navigate
cd RL-Assignment

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- `gymnasium` — OpenAI Gym-compatible environment API
- `text-flappy-bird-gym` — Text Flappy Bird environment
- `numpy`, `matplotlib`, `seaborn` — Computation & visualization
- `tqdm` — Progress bars
- `flappy-bird-gymnasium` — Original Flappy Bird (for comparison experiment)

## 📓 Usage

### Running the Notebook

Open `notebook.ipynb` in Jupyter or Google Colab. The notebook contains:

1. **Environment Exploration** — Observation ranges, reward structure
2. **MC Control Training** — Agent 1 implementation and results
3. **True Online Sarsa(λ) Training** — Agent 2 implementation and results
4. **Comparative Analysis** — Learning curves, value functions, parameter sweeps
5. **Additional Experiments** — Screen-v0, generalization, original Flappy Bird

### Using the Agents Standalone

```python
import gymnasium as gym
import text_flappy_bird_gym
from src.agents.mc_agent import MCControlAgent
from src.agents.sarsa_lambda import TrueOnlineSarsaLambda

# Create environment
env = gym.make('TextFlappyBird-v0', height=15, width=20, pipe_gap=4)

# Train MC agent
mc = MCControlAgent(alpha=0.1, gamma=0.99)
for ep in range(50000):
    mc.train_episode(env)

# Train Sarsa(λ) agent
sarsa = TrueOnlineSarsaLambda(alpha=0.025, gamma=0.99, lam=0.9)
for ep in range(50000):
    sarsa.train_episode(env)
```

## 🧠 Algorithms

### Monte Carlo Control
- **Type:** Model-free, off-line (updates after full episodes)
- **State representation:** Integer tuples (Δx, Δy) used directly as keys
- **Exploration:** ε-greedy with exponential decay
- **Update:** Q(s,a) ← Q(s,a) + α(G - Q(s,a)) using first-visit returns

### True Online Sarsa(λ)
- **Type:** Model-free, online (updates every step)
- **State representation:** Tile coding (8 tilings, 8 tiles/dim)
- **Exploration:** ε-greedy with exponential decay
- **Update:** Dutch eligibility traces for true online forward-view equivalence

## 📊 Key Hyperparameters

| Parameter | MC Control | Sarsa(λ) |
|---|---|---|
| α (learning rate) | 0.1 | 0.025 (= 0.2/8) |
| γ (discount) | 0.99 | 0.99 |
| λ (trace decay) | N/A | 0.9 |
| ε decay | 0.9999 | 0.9999 |
| ε min | 0.05 | 0.05 |

## 📚 References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.)
- Text Flappy Bird Gym: [GitLab](https://gitlab-research.centralesupelec.fr/stergios.christodoulidis/text-flappy-bird-gym)
- Flappy Bird Gymnasium: [GitHub](https://github.com/markub3327/flappy-bird-gymnasium)
