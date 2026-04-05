"""
Parameter sweep script for Sarsa(λ) and MC agents.

Runs a grid search over key hyperparameters and saves results.
Supports both sequential and parallel execution (using multiprocessing).
Designed to be run as a SLURM job array or as a single parallel job.

Usage:
    # Run all configs in parallel with 4 workers:
    python experiments/run_sweep.py --agent sarsa --num_episodes 30000 --n_workers 4

    # Run a single config (for SLURM array):
    python experiments/run_sweep.py --agent sarsa --config_id 5 --num_episodes 30000
"""

import sys, os
import argparse
import numpy as np
import json
import time
import itertools
from multiprocessing import Pool
from functools import partial

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import text_flappy_bird_gym
from src.agents.mc_agent import MCControlAgent
from src.agents.sarsa_lambda import TrueOnlineSarsaLambda


# ── Sweep configurations ──────────────────────────────────────────────

SARSA_SWEEP = {
    'alpha_base': [0.1, 0.2, 0.4, 0.8],       # will be divided by num_tilings
    'lam': [0.0, 0.4, 0.8, 0.9, 0.95, 1.0],
}

MC_SWEEP = {
    'alpha': [0.01, 0.05, 0.1, 0.2, 0.5],
    'gamma': [0.9, 0.95, 0.99, 1.0],
}


def get_sarsa_configs():
    """Generate all Sarsa parameter combinations."""
    configs = []
    for alpha_base, lam in itertools.product(
        SARSA_SWEEP['alpha_base'], SARSA_SWEEP['lam']
    ):
        configs.append({
            'alpha_base': alpha_base,
            'lam': lam,
        })
    return configs


def get_mc_configs():
    """Generate all MC parameter combinations."""
    configs = []
    for alpha, gamma in itertools.product(
        MC_SWEEP['alpha'], MC_SWEEP['gamma']
    ):
        configs.append({
            'alpha': alpha,
            'gamma': gamma,
        })
    return configs


def run_sarsa_config(config, num_episodes, seeds, env_kwargs):
    """Run Sarsa(λ) with a specific config across multiple seeds."""
    num_tilings = 8
    alpha = config['alpha_base'] / num_tilings
    lam = config['lam']
    
    seed_results = []
    
    for seed in seeds:
        np.random.seed(seed)
        env = gym.make('TextFlappyBird-v0', **env_kwargs)
        
        agent = TrueOnlineSarsaLambda(
            alpha=alpha, gamma=0.99, lam=lam,
            epsilon_decay=0.9999, epsilon_min=0.05,
            num_tilings=num_tilings, num_tiles=8, iht_size=4096
        )
        
        rewards = []
        for ep in range(num_episodes):
            metrics = agent.train_episode(env)
            rewards.append(metrics['reward'])
        
        # Evaluate (cap at 3000 steps to prevent indefinite runs with good policies)
        agent.epsilon = 0.0
        eval_rewards = []
        for _ in range(100):
            obs, _ = env.reset()
            done = False
            total_r = 0
            steps = 0
            while not done and steps < 3000:
                action = agent.greedy_action(np.array(obs, dtype=float))
                obs, r, term, trunc, _ = env.step(action)
                done = term or trunc
                total_r += r
                steps += 1
            eval_rewards.append(total_r)
        
        seed_results.append({
            'seed': seed,
            'final_avg_reward_1k': float(np.mean(rewards[-1000:])),
            'eval_mean': float(np.mean(eval_rewards)),
            'eval_std': float(np.std(eval_rewards)),
            'rewards': rewards,  # full history for learning curves
        })
        
        env.close()
    
    return {
        'alpha_base': config['alpha_base'],
        'alpha': alpha,
        'lam': lam,
        'mean_eval': float(np.mean([r['eval_mean'] for r in seed_results])),
        'std_eval': float(np.std([r['eval_mean'] for r in seed_results])),
        'seeds': seed_results,
    }


def run_mc_config(config, num_episodes, seeds, env_kwargs):
    """Run MC Control with a specific config across multiple seeds."""
    alpha = config['alpha']
    gamma = config['gamma']
    
    seed_results = []
    
    for seed in seeds:
        np.random.seed(seed)
        env = gym.make('TextFlappyBird-v0', **env_kwargs)
        
        agent = MCControlAgent(
            alpha=alpha, gamma=gamma,
            epsilon_decay=0.9999, epsilon_min=0.05
        )
        
        rewards = []
        for ep in range(num_episodes):
            metrics = agent.train_episode(env)
            rewards.append(metrics['reward'])
        
        # Evaluate (cap at 3000 steps to prevent indefinite runs with good policies)
        agent.epsilon = 0.0
        eval_rewards = []
        for _ in range(100):
            obs, _ = env.reset()
            done = False
            total_r = 0
            steps = 0
            while not done and steps < 3000:
                state = agent.get_state(obs)
                action = agent.greedy_action(state)
                obs, r, term, trunc, _ = env.step(action)
                done = term or trunc
                total_r += r
                steps += 1
            eval_rewards.append(total_r)
        
        seed_results.append({
            'seed': seed,
            'final_avg_reward_1k': float(np.mean(rewards[-1000:])),
            'eval_mean': float(np.mean(eval_rewards)),
            'eval_std': float(np.std(eval_rewards)),
            'rewards': rewards,
        })
        
        env.close()
    
    return {
        'alpha': alpha,
        'gamma': gamma,
        'mean_eval': float(np.mean([r['eval_mean'] for r in seed_results])),
        'std_eval': float(np.std([r['eval_mean'] for r in seed_results])),
        'seeds': seed_results,
    }


def run_config(agent_type, config, num_episodes, seeds, env_kwargs):
    """Wrapper to run a single configuration for either agent."""
    if agent_type == 'sarsa':
        return run_sarsa_config(config, num_episodes, seeds, env_kwargs)
    else:
        return run_mc_config(config, num_episodes, seeds, env_kwargs)


def main():
    parser = argparse.ArgumentParser(description='Parameter sweep for RL agents')
    parser.add_argument('--agent', type=str, required=True, choices=['mc', 'sarsa'])
    parser.add_argument('--num_episodes', type=int, default=30000)
    parser.add_argument('--n_seeds', type=int, default=3)
    parser.add_argument('--config_id', type=int, default=-1,
                        help='Run a single config by ID (-1 = run all)')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='Number of parallel workers (only used if config_id is -1)')
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    seeds = list(range(args.n_seeds))
    env_kwargs = dict(height=15, width=20, pipe_gap=4)
    
    if args.agent == 'sarsa':
        configs = get_sarsa_configs()
    else:
        configs = get_mc_configs()
    
    print(f"Total configurations: {len(configs)}")
    
    if args.config_id >= 0:
        # Run single config (for SLURM array)
        if args.config_id >= len(configs):
            print(f"Config ID {args.config_id} out of range (max {len(configs)-1})")
            sys.exit(1)
        
        config = configs[args.config_id]
        print(f"[{args.agent}] Running config {args.config_id}: {config}")
        
        t0 = time.time()
        result = run_config(args.agent, config, args.num_episodes, seeds, env_kwargs)
        elapsed = time.time() - t0
        
        print(f"  Mean eval: {result['mean_eval']:.1f} ± {result['std_eval']:.1f}")
        print(f"  Time: {elapsed:.1f}s")
        
        out_path = os.path.join(args.output_dir, 
                                f'sweep_{args.agent}_config{args.config_id}.json')
        # Don't save full reward histories for sweeps (too large)
        save_result = {k: v for k, v in result.items()}
        if 'seeds' in save_result:
            for s in save_result['seeds']:
                s.pop('rewards', None) # clear to save space
        
        with open(out_path, 'w') as f:
            json.dump(save_result, f)
        print(f"  Saved to {out_path}")
    
    else:
        # Run all configs
        t0 = time.time()
        print(f"[{args.agent}] Running all {len(configs)} configs using {args.n_workers} workers...")
        
        if args.n_workers > 1:
            # Parallel execution
            worker_func = partial(run_config, args.agent, num_episodes=args.num_episodes, 
                                 seeds=seeds, env_kwargs=env_kwargs)
            
            with Pool(args.n_workers) as p:
                all_results = p.map(worker_func, configs)
        else:
            # Sequential execution
            all_results = []
            for i, config in enumerate(configs):
                print(f"  [{i+1}/{len(configs)}] Config: {config}")
                res = run_config(args.agent, config, args.num_episodes, seeds, env_kwargs)
                all_results.append(res)
        
        # Cleanup histories before saving
        for res in all_results:
            if 'seeds' in res:
                for s in res['seeds']:
                    s.pop('rewards', None)
        
        elapsed = time.time() - t0
        out_path = os.path.join(args.output_dir, f'sweep_{args.agent}_all.json')
        with open(out_path, 'w') as f:
            json.dump(all_results, f)
        
        print(f"\n✅ All sweep results for {args.agent} saved to {out_path}")
        print(f"Total time: {elapsed/60:.1f} minutes")


if __name__ == '__main__':
    main()
