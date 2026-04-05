"""
Experiment runner for RL Assignment.

Provides functions for:
- Training agents with progress tracking
- Running parameter sweeps
- Multi-seed experiments
- Evaluation episodes
"""

import numpy as np
import gymnasium as gym
from tqdm import tqdm
from typing import Dict, List, Tuple, Callable, Optional
import time
import copy


def train_agent(agent, env, num_episodes: int, 
                eval_every: int = 1000,
                eval_episodes: int = 10,
                verbose: bool = True) -> Dict:
    """Train an agent and track metrics.
    
    Args:
        agent: Agent with train_episode() method
        env: Gymnasium environment
        num_episodes: Total training episodes
        eval_every: Evaluate every N episodes
        eval_episodes: Number of evaluation episodes
        verbose: Whether to print progress
        
    Returns:
        Dictionary with training history
    """
    history = {
        'rewards': [],
        'lengths': [],
        'epsilons': [],
        'eval_rewards': [],
        'eval_episodes': [],
    }
    
    pbar = tqdm(range(num_episodes), disable=not verbose, 
                desc='Training', unit='ep')
    
    for episode in pbar:
        metrics = agent.train_episode(env)
        
        history['rewards'].append(metrics['reward'])
        history['lengths'].append(metrics['length'])
        history['epsilons'].append(metrics['epsilon'])
        
        # Periodic evaluation
        if eval_every and (episode + 1) % eval_every == 0:
            eval_reward = evaluate_agent(agent, env, eval_episodes)
            history['eval_rewards'].append(eval_reward)
            history['eval_episodes'].append(episode + 1)
            
            if verbose:
                recent_rewards = history['rewards'][-100:]
                avg_reward = np.mean(recent_rewards)
                avg_length = np.mean(history['lengths'][-100:])
                pbar.set_postfix({
                    'avg_R': f'{avg_reward:.1f}',
                    'avg_L': f'{avg_length:.1f}',
                    'eval': f'{eval_reward:.1f}',
                    'ε': f'{metrics["epsilon"]:.3f}'
                })
    
    return history


def evaluate_agent(agent, env, num_episodes: int = 10) -> float:
    """Evaluate agent performance without exploration.
    
    Args:
        agent: Agent with greedy_action() or select_action() method
        env: Gymnasium environment
        num_episodes: Number of evaluation episodes
        
    Returns:
        Average total reward across evaluation episodes
    """
    total_rewards = []
    
    # Temporarily disable exploration
    original_epsilon = getattr(agent, 'epsilon', None)
    if original_epsilon is not None:
        agent.epsilon = 0.0
    
    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            # Use different method depending on agent type
            if hasattr(agent, 'get_state'):
                # MC agent: discretize first
                state = agent.get_state(obs)
                action = agent.greedy_action(state)
            else:
                # Sarsa agent: use obs directly
                action = agent.greedy_action(obs)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        total_rewards.append(episode_reward)
    
    # Restore exploration
    if original_epsilon is not None:
        agent.epsilon = original_epsilon
    
    return np.mean(total_rewards)


def run_parameter_sweep(agent_class, agent_base_kwargs: Dict,
                         env_id: str, env_kwargs: Dict,
                         sweep_param1: str, values1: List,
                         sweep_param2: str = None, values2: List = None,
                         num_episodes: int = 10000,
                         eval_episodes: int = 20,
                         n_seeds: int = 3,
                         verbose: bool = True) -> Dict:
    """Run a parameter sweep experiment.
    
    Args:
        agent_class: Agent class to instantiate
        agent_base_kwargs: Base keyword arguments for the agent
        env_id: Environment ID string
        env_kwargs: Environment keyword arguments
        sweep_param1: First parameter name to sweep
        values1: Values for first parameter
        sweep_param2: Optional second parameter name
        values2: Values for second parameter
        num_episodes: Training episodes per configuration
        eval_episodes: Evaluation episodes per configuration
        n_seeds: Number of seeds per configuration
        verbose: Whether to print progress
        
    Returns:
        Dictionary with sweep results
    """
    if values2 is None:
        values2 = [None]
        sweep_param2_name = None
    else:
        sweep_param2_name = sweep_param2
    
    results = np.zeros((len(values1), len(values2)))
    all_histories = {}
    
    total = len(values1) * len(values2) * n_seeds
    pbar = tqdm(total=total, disable=not verbose, desc='Parameter Sweep')
    
    for i, v1 in enumerate(values1):
        for j, v2 in enumerate(values2):
            seed_rewards = []
            
            for seed in range(n_seeds):
                np.random.seed(seed * 42)
                
                # Create agent with swept parameters
                kwargs = agent_base_kwargs.copy()
                kwargs[sweep_param1] = v1
                if sweep_param2_name:
                    kwargs[sweep_param2_name] = v2
                
                agent = agent_class(**kwargs)
                env = gym.make(env_id, **env_kwargs)
                
                # Train
                history = train_agent(agent, env, num_episodes, 
                                      eval_every=0, verbose=False)
                
                # Evaluate
                eval_reward = evaluate_agent(agent, env, eval_episodes)
                seed_rewards.append(eval_reward)
                
                env.close()
                pbar.update(1)
            
            results[i, j] = np.mean(seed_rewards)
            
            config_key = f'{sweep_param1}={v1}'
            if sweep_param2_name:
                config_key += f', {sweep_param2_name}={v2}'
            all_histories[config_key] = seed_rewards
            
            if verbose:
                pbar.set_postfix({
                    sweep_param1: f'{v1}', 
                    'score': f'{results[i, j]:.1f}'
                })
    
    pbar.close()
    
    return {
        'results': results,
        'param1_values': values1,
        'param2_values': values2 if sweep_param2_name else None,
        'param1_name': sweep_param1,
        'param2_name': sweep_param2_name,
        'all_histories': all_histories,
    }


def run_multi_seed(agent_class, agent_kwargs: Dict,
                    env_id: str, env_kwargs: Dict,
                    num_episodes: int = 50000,
                    n_seeds: int = 5,
                    verbose: bool = True) -> Dict:
    """Run training with multiple seeds for confidence intervals.
    
    Args:
        agent_class: Agent class
        agent_kwargs: Agent keyword arguments
        env_id: Environment ID
        env_kwargs: Environment kwargs
        num_episodes: Episodes per seed
        n_seeds: Number of seeds
        verbose: Whether to print progress
        
    Returns:
        Dictionary with per-seed histories
    """
    all_rewards = []
    all_lengths = []
    
    for seed in range(n_seeds):
        if verbose:
            print(f"\n--- Seed {seed + 1}/{n_seeds} ---")
        
        np.random.seed(seed * 42)
        agent = agent_class(**agent_kwargs)
        env = gym.make(env_id, **env_kwargs)
        
        history = train_agent(agent, env, num_episodes, verbose=verbose)
        all_rewards.append(history['rewards'])
        all_lengths.append(history['lengths'])
        
        env.close()
    
    return {
        'rewards': all_rewards,
        'lengths': all_lengths,
    }


def explore_environment(env_id: str, env_kwargs: Dict = None,
                         num_episodes: int = 100) -> Dict:
    """Explore an environment to determine observation ranges.
    
    Args:
        env_id: Environment ID
        env_kwargs: Environment kwargs
        num_episodes: Number of random episodes to run
        
    Returns:
        Dictionary with observation statistics
    """
    if env_kwargs is None:
        env_kwargs = {}
    
    env = gym.make(env_id, **env_kwargs)
    
    all_obs = []
    all_rewards = []
    episode_lengths = []
    
    for _ in range(num_episodes):
        obs, info = env.reset()
        all_obs.append(obs)
        done = False
        ep_len = 0
        
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            all_obs.append(obs)
            all_rewards.append(reward)
            ep_len += 1
        
        episode_lengths.append(ep_len)
    
    all_obs = np.array(all_obs)
    
    env.close()
    
    stats = {
        'obs_min': all_obs.min(axis=0),
        'obs_max': all_obs.max(axis=0),
        'obs_mean': all_obs.mean(axis=0),
        'obs_std': all_obs.std(axis=0),
        'obs_shape': all_obs.shape[1:],
        'reward_min': min(all_rewards),
        'reward_max': max(all_rewards),
        'reward_mean': np.mean(all_rewards),
        'avg_episode_length': np.mean(episode_lengths),
        'max_episode_length': max(episode_lengths),
        'min_episode_length': min(episode_lengths),
        'num_observations': len(all_obs),
    }
    
    return stats
