"""
True Online Sarsa(λ) Agent with Tile Coding.

Implements the True Online Sarsa(λ) algorithm from Section 12.7 of
Sutton & Barto's "Reinforcement Learning: An Introduction" (2nd ed.).

Uses linear function approximation with tile coding features:
    q̂(s, a, w) = w⊤ x(s, a)
"""

import numpy as np
from typing import Dict, Tuple, List, Optional

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.tile_coding import TileCoder


class TrueOnlineSarsaLambda:
    """True Online Sarsa(λ) with linear function approximation.
    
    This agent uses eligibility traces with the "dutch trace" update
    to achieve online performance equivalent to the offline λ-return
    algorithm, while maintaining O(d) per-step complexity.
    
    Key equations:
        δ = R + γ q' - q
        z ← γλz + (1 - αγλ z⊤x) x         (dutch trace)
        w ← w + α(δ + q - q_old)z - α(q - q_old)x
        q_old ← q'
    """
    
    def __init__(self,
                 n_actions: int = 2,
                 alpha: float = 0.025,  # α / num_tilings
                 gamma: float = 0.99,
                 lam: float = 0.9,     # λ (trace decay)
                 epsilon_start: float = 1.0,
                 epsilon_decay: float = 0.9999,
                 epsilon_min: float = 0.05,
                 num_tilings: int = 8,
                 num_tiles: int = 8,
                 iht_size: int = 4096,
                 obs_low: np.ndarray = None,
                 obs_high: np.ndarray = None):
        """
        Args:
            n_actions: Number of actions
            alpha: Step size (should be ~ base_alpha / num_tilings)
            gamma: Discount factor
            lam: Trace decay parameter λ
            epsilon_start: Initial exploration rate
            epsilon_decay: Multiplicative decay per episode
            epsilon_min: Minimum exploration rate
            num_tilings: Number of tile coding tilings
            num_tiles: Number of tiles per dimension per tiling
            iht_size: Index hash table size
            obs_low: Lower bounds of observation space
            obs_high: Upper bounds of observation space
        """
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Tile coder for feature construction
        self.tile_coder = TileCoder(
            num_tilings=num_tilings,
            num_tiles=num_tiles,
            iht_size=iht_size,
            obs_low=obs_low,
            obs_high=obs_high
        )
        
        # Weight vector
        self.w = np.zeros(iht_size)
        
        # Feature size
        self.d = iht_size
        self.num_tilings = num_tilings
        
        # Training history
        self.episode_rewards = []
        self.episode_lengths = []
        self.epsilon_history = []
        
        # Episode-level variables (reset each episode)
        self._z = None      # Eligibility trace
        self._q_old = None  # Previous q value
        self._x = None      # Current feature vector
        self._action = None # Current action
    
    def _get_q(self, obs: np.ndarray, action: int) -> float:
        """Compute q̂(s, a, w) = w⊤ x(s, a)."""
        active_tiles = self.tile_coder.get_tiles(obs, action)
        q = 0.0
        for idx in active_tiles:
            if idx is not None:
                q += self.w[idx]
        return q
    
    def _get_x_sparse(self, obs: np.ndarray, action: int) -> List[int]:
        """Get active tile indices (sparse representation)."""
        return self.tile_coder.get_tiles(obs, action)
    
    def select_action(self, obs: np.ndarray) -> int:
        """Select action using ε-greedy policy.
        
        Args:
            obs: Continuous observation [Δx, Δy]
            
        Returns:
            Selected action
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        # Compute Q for all actions, pick best
        q_values = [self._get_q(obs, a) for a in range(self.n_actions)]
        return int(np.argmax(q_values))
    
    def greedy_action(self, obs: np.ndarray) -> int:
        """Select action greedily (for evaluation)."""
        q_values = [self._get_q(obs, a) for a in range(self.n_actions)]
        return int(np.argmax(q_values))
    
    def start_episode(self, obs: np.ndarray) -> int:
        """Initialize episode variables and select first action.
        
        Args:
            obs: Initial observation
            
        Returns:
            First action
        """
        action = self.select_action(obs)
        
        # Initialize eligibility trace
        self._z = np.zeros(self.d)
        self._q_old = 0.0
        self._x = self._get_x_sparse(obs, action)
        self._action = action
        
        return action
    
    def step(self, obs: np.ndarray, reward: float, 
             next_obs: np.ndarray, done: bool) -> int:
        """Perform one True Online Sarsa(λ) update step.
        
        Delegates to _step_impl which follows the pseudocode exactly.
        
        Args:
            obs: Current observation
            reward: Received reward
            next_obs: Next observation
            done: Whether episode is terminal
            
        Returns:
            Next action (or -1 if terminal)
        """
        return self._step_impl(reward, next_obs, done)
    
    def _step_impl(self, reward: float, next_obs: np.ndarray, done: bool) -> int:
        """True Online Sarsa(λ) step implementation.
        
        Following the pseudocode exactly:
            q  ← w⊤x
            q' ← w⊤x'  (0 if terminal)
            δ  ← R + γq' - q
            z  ← γλz + (1 - αγλ z⊤x)x
            w  ← w + α(δ + q - q_old)z - α(q - q_old)x
            q_old ← q'
        """
        x_indices = self._x
        
        # q = w⊤x (using sparse representation)
        q = sum(self.w[idx] for idx in x_indices if idx is not None)
        
        if done:
            q_prime = 0.0
            next_action = -1
            x_prime_indices = []
        else:
            next_action = self.select_action(next_obs)
            x_prime_indices = self._get_x_sparse(next_obs, next_action)
            q_prime = sum(self.w[idx] for idx in x_prime_indices if idx is not None)
        
        # TD error
        delta = reward + self.gamma * q_prime - q
        
        # Dutch trace update: z ← γλz + (1 - αγλ z⊤x)x
        # First compute z⊤x
        z_dot_x = sum(self._z[idx] for idx in x_indices if idx is not None)
        
        # Decay all traces
        self._z *= self.gamma * self.lam
        
        # Add contribution for active tiles
        coeff = 1.0 - self.alpha * self.gamma * self.lam * z_dot_x
        for idx in x_indices:
            if idx is not None:
                self._z[idx] += coeff
        
        # Weight update: w ← w + α(δ + q - q_old)z - α(q - q_old)x
        update_coeff_z = self.alpha * (delta + q - self._q_old)
        update_coeff_x = self.alpha * (q - self._q_old)
        
        self.w += update_coeff_z * self._z
        for idx in x_indices:
            if idx is not None:
                self.w[idx] -= update_coeff_x
        
        # Update for next step
        self._q_old = q_prime
        self._x = x_prime_indices
        self._action = next_action
        
        return next_action
    
    def train_episode(self, env) -> Dict:
        """Run one full training episode.
        
        Args:
            env: Gymnasium environment
            
        Returns:
            Dictionary with episode metrics
        """
        obs, info = env.reset()
        action = self.start_episode(obs)
        
        total_reward = 0.0
        steps = 0
        done = False
        
        while not done:
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            action = self._step_impl(reward, next_obs, done)
        
        # Track metrics
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(steps)
        self.epsilon_history.append(self.epsilon)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return {
            'reward': total_reward,
            'length': steps,
            'epsilon': self.epsilon,
        }
    
    def get_value_grid(self, x_range: Tuple[float, float] = (0, 25),
                       y_range: Tuple[float, float] = (-15, 15),
                       resolution: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute value function and policy on a grid for visualization.
        
        Args:
            x_range: Range for Δx axis
            y_range: Range for Δy axis
            resolution: Grid resolution per dimension
            
        Returns:
            (x_grid, y_grid, V_grid, policy_grid)
        """
        x_vals = np.linspace(x_range[0], x_range[1], resolution)
        y_vals = np.linspace(y_range[0], y_range[1], resolution)
        
        V_grid = np.zeros((resolution, resolution))
        policy_grid = np.zeros((resolution, resolution))
        
        for i, x in enumerate(x_vals):
            for j, y in enumerate(y_vals):
                obs = np.array([x, y])
                q_values = [self._get_q(obs, a) for a in range(self.n_actions)]
                V_grid[i, j] = max(q_values)
                policy_grid[i, j] = np.argmax(q_values)
        
        return x_vals, y_vals, V_grid, policy_grid
    
    def save(self, filepath: str):
        """Save weights to file."""
        np.savez(filepath,
                 w=self.w,
                 epsilon=self.epsilon,
                 episode_rewards=self.episode_rewards,
                 episode_lengths=self.episode_lengths)
