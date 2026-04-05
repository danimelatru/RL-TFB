"""
Main training script — trains both MC Control and Sarsa(λ) agents.

Saves results (rewards, lengths, Q-tables/weights) to results/ directory.
Designed to be run on the HPC cluster via SLURM or locally.

Usage:
    python experiments/run_training.py [--num_episodes N] [--seed S]
"""

import sys, os
import argparse
import numpy as np
import json
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import text_flappy_bird_gym
from src.agents.mc_agent import MCControlAgent
from src.agents.sarsa_lambda import TrueOnlineSarsaLambda


def train_mc(env, num_episodes, seed, alpha=0.1, gamma=0.99,
             eps_decay=0.9999, eps_min=0.05):
    """Train MC Control agent and return results."""
    np.random.seed(seed)
    
    agent = MCControlAgent(
        alpha=alpha, gamma=gamma,
        epsilon_decay=eps_decay, epsilon_min=eps_min
    )
    
    rewards = []
    lengths = []
    
    for ep in range(num_episodes):
        metrics = agent.train_episode(env)
        rewards.append(metrics['reward'])
        lengths.append(metrics['length'])
        
        if (ep + 1) % 10000 == 0:
            avg_r = np.mean(rewards[-1000:])
            print(f"  MC [{ep+1}/{num_episodes}] avg_reward(1k)={avg_r:.1f} eps={agent.epsilon:.4f}")
    
    return agent, rewards, lengths


def train_sarsa(env, num_episodes, seed, alpha=0.025, gamma=0.99,
                lam=0.9, eps_decay=0.9999, eps_min=0.05,
                num_tilings=8, num_tiles=8, iht_size=4096):
    """Train True Online Sarsa(λ) agent and return results."""
    np.random.seed(seed)
    
    agent = TrueOnlineSarsaLambda(
        alpha=alpha, gamma=gamma, lam=lam,
        epsilon_decay=eps_decay, epsilon_min=eps_min,
        num_tilings=num_tilings, num_tiles=num_tiles,
        iht_size=iht_size
    )
    
    rewards = []
    lengths = []
    
    for ep in range(num_episodes):
        metrics = agent.train_episode(env)
        rewards.append(metrics['reward'])
        lengths.append(metrics['length'])
        
        if (ep + 1) % 10000 == 0:
            avg_r = np.mean(rewards[-1000:])
            print(f"  Sarsa [{ep+1}/{num_episodes}] avg_reward(1k)={avg_r:.1f} eps={agent.epsilon:.4f}")
    
    return agent, rewards, lengths


def evaluate(agent, env, num_episodes=50, is_mc=True):
    """Evaluate agent greedily."""
    old_eps = agent.epsilon
    agent.epsilon = 0.0
    
    eval_rewards = []
    eval_lengths = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_r = 0
        steps = 0
        
        while not done:
            if is_mc:
                state = agent.get_state(obs)
                action = agent.greedy_action(state)
            else:
                action = agent.greedy_action(np.array(obs, dtype=float))
            
            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            total_r += r
            steps += 1
        
        eval_rewards.append(total_r)
        eval_lengths.append(steps)
    
    agent.epsilon = old_eps
    return eval_rewards, eval_lengths


def main():
    parser = argparse.ArgumentParser(description='Train RL agents on Text Flappy Bird')
    parser.add_argument('--num_episodes', type=int, default=50000, help='Training episodes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--agent', type=str, default='both', choices=['mc', 'sarsa', 'both'],
                        help='Which agent to train')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    
    # MC hyperparameters
    parser.add_argument('--mc_alpha', type=float, default=0.1)
    parser.add_argument('--mc_gamma', type=float, default=0.99)
    
    # Sarsa hyperparameters
    parser.add_argument('--sarsa_alpha', type=float, default=0.025)
    parser.add_argument('--sarsa_gamma', type=float, default=0.99)
    parser.add_argument('--sarsa_lambda', type=float, default=0.9)
    parser.add_argument('--sarsa_tilings', type=int, default=8)
    parser.add_argument('--sarsa_tiles', type=int, default=8)
    parser.add_argument('--sarsa_iht', type=int, default=4096)
    
    # Environment
    parser.add_argument('--height', type=int, default=15)
    parser.add_argument('--width', type=int, default=20)
    parser.add_argument('--pipe_gap', type=int, default=4)
    
    # Epsilon schedule
    parser.add_argument('--eps_decay', type=float, default=0.9999)
    parser.add_argument('--eps_min', type=float, default=0.05)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
    
    env_kwargs = dict(height=args.height, width=args.width, pipe_gap=args.pipe_gap)
    env = gym.make('TextFlappyBird-v0', **env_kwargs)
    
    config_tag = f"h{args.height}_w{args.width}_g{args.pipe_gap}_seed{args.seed}"
    
    results = {
        'config': vars(args),
        'env_kwargs': env_kwargs,
    }
    
    # ── Train MC ───────────────────────────────────────────────────
    if args.agent in ['mc', 'both']:
        print(f"\n{'='*60}")
        print(f"Training MC Control (seed={args.seed}, episodes={args.num_episodes})")
        print(f"{'='*60}")
        
        t0 = time.time()
        mc_agent, mc_rewards, mc_lengths = train_mc(
            env, args.num_episodes, args.seed,
            alpha=args.mc_alpha, gamma=args.mc_gamma,
            eps_decay=args.eps_decay, eps_min=args.eps_min
        )
        mc_time = time.time() - t0
        
        # Evaluate
        mc_eval_rewards, mc_eval_lengths = evaluate(mc_agent, env, 50, is_mc=True)
        
        print(f"\nMC Training complete in {mc_time:.1f}s")
        print(f"  Eval reward: {np.mean(mc_eval_rewards):.1f} ± {np.std(mc_eval_rewards):.1f}")
        print(f"  Q-table size: {len(mc_agent.Q)} states")
        
        results['mc'] = {
            'rewards': mc_rewards,
            'lengths': mc_lengths,
            'eval_rewards': mc_eval_rewards,
            'eval_lengths': mc_eval_lengths,
            'train_time': mc_time,
            'q_table_size': len(mc_agent.Q),
        }
        
        # Save Q-table
        q_data = {str(k): v.tolist() for k, v in mc_agent.Q.items()}
        mc_path = os.path.join(args.output_dir, 'models', f'mc_q_table_{config_tag}.json')
        with open(mc_path, 'w') as f:
            json.dump(q_data, f)
        print(f"  Q-table saved to {mc_path}")
    
    # ── Train Sarsa(λ) ────────────────────────────────────────────
    if args.agent in ['sarsa', 'both']:
        print(f"\n{'='*60}")
        print(f"Training Sarsa(λ) (seed={args.seed}, episodes={args.num_episodes})")
        print(f"  α={args.sarsa_alpha}, λ={args.sarsa_lambda}, tilings={args.sarsa_tilings}")
        print(f"{'='*60}")
        
        t0 = time.time()
        sarsa_agent, sarsa_rewards, sarsa_lengths = train_sarsa(
            env, args.num_episodes, args.seed,
            alpha=args.sarsa_alpha, gamma=args.sarsa_gamma,
            lam=args.sarsa_lambda, eps_decay=args.eps_decay, eps_min=args.eps_min,
            num_tilings=args.sarsa_tilings, num_tiles=args.sarsa_tiles,
            iht_size=args.sarsa_iht
        )
        sarsa_time = time.time() - t0
        
        # Evaluate
        sarsa_eval_rewards, sarsa_eval_lengths = evaluate(
            sarsa_agent, env, 50, is_mc=False
        )
        
        print(f"\nSarsa(λ) Training complete in {sarsa_time:.1f}s")
        print(f"  Eval reward: {np.mean(sarsa_eval_rewards):.1f} ± {np.std(sarsa_eval_rewards):.1f}")
        print(f"  Non-zero weights: {np.count_nonzero(sarsa_agent.w)}/{len(sarsa_agent.w)}")
        
        results['sarsa'] = {
            'rewards': sarsa_rewards,
            'lengths': sarsa_lengths,
            'eval_rewards': sarsa_eval_rewards,
            'eval_lengths': sarsa_eval_lengths,
            'train_time': sarsa_time,
            'nonzero_weights': int(np.count_nonzero(sarsa_agent.w)),
        }
        
        # Save weights
        w_path = os.path.join(args.output_dir, 'models', f'sarsa_weights_{config_tag}.npz')
        np.savez(w_path, w=sarsa_agent.w)
        print(f"  Weights saved to {w_path}")
    
    env.close()
    
    # ── Save all results ──────────────────────────────────────────
    results_path = os.path.join(args.output_dir, f'training_results_{config_tag}.json')
    
    # Convert numpy arrays to lists for JSON serialization
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(results_path, 'w') as f:
        json.dump(results, f, default=convert)
    
    print(f"\n✅ All results saved to {results_path}")


if __name__ == '__main__':
    main()
