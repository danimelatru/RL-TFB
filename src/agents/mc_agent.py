"""
Monte Carlo Control Agent with ε-greedy exploration.

Implements constant-α first-visit MC Control for the TextFlappyBird-v0 
environment. For TextFlappyBird-v0, observations are already integer tuples
(Δx, Δy) and can be used directly as state keys. An optional discretizer
is available for environments with continuous observations.
"""

import numpy as np
from collections import defaultdict
from typing import Tuple, Dict, List, Optional


class StateDiscretizer:
    """Discretizes continuous (Δx, Δy) observations into bins."""
    
    def __init__(self, num_x_bins: int = 10, num_y_bins: int = 10,
                 x_range: Tuple[float, float] = (0, 25),
                 y_range: Tuple[float, float] = (-15, 15)):
        """
        Args:
            num_x_bins: Number of bins for the x (horizontal distance) axis
            num_y_bins: Number of bins for the y (vertical distance) axis
            x_range: (min, max) range for Δx
            y_range: (min, max) range for Δy
        """
        self.num_x_bins = num_x_bins
        self.num_y_bins = num_y_bins
        self.x_bins = np.linspace(x_range[0], x_range[1], num_x_bins + 1)
        self.y_bins = np.linspace(y_range[0], y_range[1], num_y_bins + 1)
    
    def discretize(self, obs: np.ndarray) -> Tuple[int, int]:
        """Convert continuous observation to discrete state.
        
        Args:
            obs: Observation array [Δx, Δy]
            
        Returns:
            Discrete state tuple (x_bin, y_bin)
        """
        x_bin = np.clip(np.digitize(obs[0], self.x_bins) - 1, 0, self.num_x_bins - 1)
        y_bin = np.clip(np.digitize(obs[1], self.y_bins) - 1, 0, self.num_y_bins - 1)
        return (int(x_bin), int(y_bin))
    
    @property
    def num_states(self) -> int:
        return self.num_x_bins * self.num_y_bins


class MCControlAgent:
    """Constant-α First-Visit Monte Carlo Control with ε-greedy policy.
    
    This agent:
    1. Generates full episodes using ε-greedy action selection
    2. Computes returns G for each first-visited (state, action) pair
    3. Updates Q-values using constant learning rate: Q(s,a) ← Q(s,a) + α(G - Q(s,a))
    
    For TextFlappyBird-v0, observations are integer tuples (Δx, Δy) which
    are used directly as state keys. For environments with continuous
    observations, set use_discretizer=True.
    """
    
    def __init__(self, 
                 n_actions: int = 2,
                 alpha: float = 0.1,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_decay: float = 0.9999,
                 epsilon_min: float = 0.05,
                 use_discretizer: bool = False,
                 num_x_bins: int = 10,
                 num_y_bins: int = 10,
                 x_range: Tuple[float, float] = (0, 14),
                 y_range: Tuple[float, float] = (-12, 12)):
        """
        Args:
            n_actions: Number of actions (2 for TFB: no-flap, flap)
            alpha: Constant learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_decay: Multiplicative decay per episode
            epsilon_min: Minimum exploration rate
            use_discretizer: Whether to bin continuous observations
            num_x_bins: Number of discretization bins for Δx
            num_y_bins: Number of discretization bins for Δy
            x_range: Observation range for Δx
            y_range: Observation range for Δy
        """
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.use_discretizer = use_discretizer
        
        # State discretizer (optional, for continuous obs environments)
        self.discretizer = StateDiscretizer(
            num_x_bins, num_y_bins, x_range, y_range
        ) if use_discretizer else None
        
        # Q-table: maps (state, action) → value
        # Using defaultdict for automatic initialization to 0
        self.Q = defaultdict(lambda: np.zeros(self.n_actions))
        
        # Training history
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_scores = []
        self.epsilon_history = []
    
    def get_state(self, obs) -> Tuple[int, int]:
        """Convert observation to discrete state.
        
        For TFB-v0, obs is already an integer tuple — use directly.
        For continuous envs with use_discretizer=True, bins the obs.
        """
        if self.use_discretizer:
            return self.discretizer.discretize(np.array(obs))
        # TFB-v0 returns integer tuples — use directly
        return tuple(int(x) for x in obs)
    
    def select_action(self, state: Tuple[int, int]) -> int:
        """Select action using ε-greedy policy.
        
        Args:
            state: Discrete state tuple
            
        Returns:
            Selected action (0 or 1)
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return int(np.argmax(self.Q[state]))
    
    def greedy_action(self, state: Tuple[int, int]) -> int:
        """Select action greedily (for evaluation)."""
        return int(np.argmax(self.Q[state]))
    
    def generate_episode(self, env) -> List[Tuple]:
        """Generate a full episode using current ε-greedy policy.
        
        Args:
            env: Gymnasium environment
            
        Returns:
            List of (state, action, reward) tuples
        """
        episode = []
        obs, info = env.reset()
        state = self.get_state(obs)
        done = False
        
        while not done:
            action = self.select_action(state)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode.append((state, action, reward))
            
            if not done:
                state = self.get_state(next_obs)
        
        return episode
    
    def update_from_episode(self, episode: List[Tuple]):
        """Update Q-values using first-visit MC with constant α.
        
        Args:
            episode: List of (state, action, reward) tuples
        """
        # Track first visits
        visited = set()
        
        # Calculate returns backwards
        G = 0.0
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = self.gamma * G + reward
            
            sa_pair = (state, action)
            
            # First-visit check
            if sa_pair not in visited:
                visited.add(sa_pair)
                # Constant-α update
                self.Q[state][action] += self.alpha * (G - self.Q[state][action])
    
    def train_episode(self, env) -> Dict:
        """Run one training episode.
        
        Args:
            env: Gymnasium environment
            
        Returns:
            Dictionary with episode metrics
        """
        # Generate episode
        episode = self.generate_episode(env)
        
        # Update Q-values
        self.update_from_episode(episode)
        
        # Compute metrics
        total_reward = sum(r for _, _, r in episode)
        ep_length = len(episode)
        
        # Track metrics
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(ep_length)
        self.epsilon_history.append(self.epsilon)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return {
            'reward': total_reward,
            'length': ep_length,
            'epsilon': self.epsilon,
        }
    
    def get_value_function(self) -> Dict:
        """Extract V(s) = max_a Q(s,a) for all visited states.
        
        Returns:
            Dictionary mapping state → value
        """
        V = {}
        for state, q_values in self.Q.items():
            V[state] = np.max(q_values)
        return V
    
    def get_policy(self) -> Dict:
        """Extract greedy policy π(s) = argmax_a Q(s,a).
        
        Returns:
            Dictionary mapping state → best action
        """
        policy = {}
        for state, q_values in self.Q.items():
            policy[state] = int(np.argmax(q_values))
        return policy
    
    def get_q_table_as_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert Q-table to 2D arrays for plotting.
        
        Returns:
            (V_grid, policy_grid, Q_grids) where:
            - V_grid: (num_x_bins, num_y_bins) array of state values
            - policy_grid: (num_x_bins, num_y_bins) array of best actions
            - Q_grids: (2, num_x_bins, num_y_bins) array of Q-values per action
        """
        nx = self.discretizer.num_x_bins
        ny = self.discretizer.num_y_bins
        
        V_grid = np.full((nx, ny), np.nan)
        policy_grid = np.full((nx, ny), np.nan)
        Q_grids = np.full((self.n_actions, nx, ny), np.nan)
        
        for (x, y), q_values in self.Q.items():
            if 0 <= x < nx and 0 <= y < ny:
                V_grid[x, y] = np.max(q_values)
                policy_grid[x, y] = np.argmax(q_values)
                for a in range(self.n_actions):
                    Q_grids[a, x, y] = q_values[a]
        
        return V_grid, policy_grid, Q_grids
    
    def save(self, filepath: str):
        """Save Q-table to file."""
        Q_dict = {str(k): v.tolist() for k, v in self.Q.items()}
        np.savez(filepath, 
                 Q_keys=list(Q_dict.keys()),
                 Q_values=list(Q_dict.values()),
                 epsilon=self.epsilon,
                 episode_rewards=self.episode_rewards,
                 episode_lengths=self.episode_lengths)
