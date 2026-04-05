"""
Generalization & cross-environment experiment script.

Tests trained agents on different TFB configurations (Q4) and
explores the screen-v0 variant (Q2).

Usage:
    python experiments/run_generalization.py --num_episodes 50000 --seed 42
"""

import sys, os
import argparse
import numpy as np
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import text_flappy_bird_gym
from src.agents.mc_agent import MCControlAgent
from src.agents.sarsa_lambda import TrueOnlineSarsaLambda


def evaluate_agent(agent, env, num_episodes=50, is_mc=True, max_steps=3000):
    """Evaluate agent greedily."""
    old_eps = agent.epsilon
    agent.epsilon = 0.0

    eval_rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_r = 0
        steps = 0
        while not done and steps < max_steps:
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

    agent.epsilon = old_eps
    return eval_rewards


def run_generalization_experiment(num_episodes, seed):
    """Q4: Train on default config, test on different configs."""
    print("\n" + "="*60)
    print("EXPERIMENT: Generalization Across Configurations (Q4)")
    print("="*60)
    
    train_config = dict(height=15, width=20, pipe_gap=4)
    test_configs = [
        dict(height=15, width=20, pipe_gap=4),   # Same (baseline)
        dict(height=15, width=20, pipe_gap=3),   # Harder (smaller gap)
        dict(height=15, width=20, pipe_gap=6),   # Easier (larger gap)
        dict(height=20, width=30, pipe_gap=4),   # Bigger screen
        dict(height=10, width=15, pipe_gap=4),   # Smaller screen
    ]
    
    results = {'train_config': train_config, 'test_configs': [], 'num_episodes': num_episodes}
    
    # Train both agents on default config
    env_train = gym.make('TextFlappyBird-v0', **train_config)
    
    # MC
    np.random.seed(seed)
    mc = MCControlAgent(alpha=0.1, gamma=0.99, epsilon_decay=0.9999, epsilon_min=0.05)
    for ep in range(num_episodes):
        mc.train_episode(env_train)
        if (ep + 1) % 10000 == 0:
            print(f"  MC training: {ep+1}/{num_episodes}")
    
    # Sarsa
    np.random.seed(seed)
    sarsa = TrueOnlineSarsaLambda(
        alpha=0.025, gamma=0.99, lam=0.9,
        epsilon_decay=0.9999, epsilon_min=0.05
    )
    for ep in range(num_episodes):
        sarsa.train_episode(env_train)
        if (ep + 1) % 10000 == 0:
            print(f"  Sarsa training: {ep+1}/{num_episodes}")
    
    env_train.close()
    
    # Test on each config
    for tc in test_configs:
        print(f"\n  Testing on config: {tc}")
        env_test = gym.make('TextFlappyBird-v0', **tc)
        
        mc_eval = evaluate_agent(mc, env_test, 100, is_mc=True)
        sarsa_eval = evaluate_agent(sarsa, env_test, 100, is_mc=False)
        
        config_result = {
            'config': tc,
            'mc_eval_mean': float(np.mean(mc_eval)),
            'mc_eval_std': float(np.std(mc_eval)),
            'sarsa_eval_mean': float(np.mean(sarsa_eval)),
            'sarsa_eval_std': float(np.std(sarsa_eval)),
        }
        results['test_configs'].append(config_result)
        
        print(f"    MC: {config_result['mc_eval_mean']:.1f} ± {config_result['mc_eval_std']:.1f}")
        print(f"    Sarsa: {config_result['sarsa_eval_mean']:.1f} ± {config_result['sarsa_eval_std']:.1f}")
        
        env_test.close()
    
    return results


def run_screen_experiment(num_episodes, seed):
    """Q2: Compare v0 vs screen-v0 environments."""
    print("\n" + "="*60)
    print("EXPERIMENT: TextFlappyBird-v0 vs screen-v0 (Q2)")
    print("="*60)
    
    env_kwargs = dict(height=15, width=20, pipe_gap=4)
    
    results = {'v0': {}, 'screen_v0': {}}
    
    # ── v0 (distance observations) ──
    print("\n  Training on TextFlappyBird-v0...")
    env_v0 = gym.make('TextFlappyBird-v0', **env_kwargs)
    
    np.random.seed(seed)
    mc_v0 = MCControlAgent(alpha=0.1, gamma=0.99, epsilon_decay=0.9999, epsilon_min=0.05)
    rewards_v0 = []
    for ep in range(num_episodes):
        m = mc_v0.train_episode(env_v0)
        rewards_v0.append(m['reward'])
    
    mc_v0_eval = evaluate_agent(mc_v0, env_v0, 100, is_mc=True)
    results['v0'] = {
        'rewards': rewards_v0,
        'eval_mean': float(np.mean(mc_v0_eval)),
        'eval_std': float(np.std(mc_v0_eval)),
        'q_table_size': len(mc_v0.Q),
    }
    print(f"    v0 eval: {results['v0']['eval_mean']:.1f} ± {results['v0']['eval_std']:.1f}")
    env_v0.close()
    
    # ── screen-v0 (full screen observations) ──
    print("\n  Training on TextFlappyBird-screen-v0...")
    env_screen = gym.make('TextFlappyBird-screen-v0', **env_kwargs)
    
    # For screen-v0, we need to discretize the screen somehow
    # Strategy: flatten + hash to create a state key
    np.random.seed(seed)
    mc_screen = MCControlAgent(
        alpha=0.1, gamma=0.99, epsilon_decay=0.9999, epsilon_min=0.05,
        use_discretizer=False  # We'll override get_state
    )
    
    # Override get_state to hash the screen
    original_get_state = mc_screen.get_state
    def screen_get_state(obs):
        # Convert screen array to a hashable tuple
        return tuple(np.array(obs).flatten().tolist())
    mc_screen.get_state = screen_get_state
    
    rewards_screen = []
    for ep in range(num_episodes):
        m = mc_screen.train_episode(env_screen)
        rewards_screen.append(m['reward'])
        if (ep + 1) % 10000 == 0:
            print(f"    screen-v0 training: {ep+1}/{num_episodes}, "
                  f"Q-table size: {len(mc_screen.Q)}")
    
    mc_screen_eval = evaluate_agent(mc_screen, env_screen, 100, is_mc=True)
    results['screen_v0'] = {
        'rewards': rewards_screen,
        'eval_mean': float(np.mean(mc_screen_eval)),
        'eval_std': float(np.std(mc_screen_eval)),
        'q_table_size': len(mc_screen.Q),
    }
    print(f"    screen-v0 eval: {results['screen_v0']['eval_mean']:.1f} ± {results['screen_v0']['eval_std']:.1f}")
    print(f"    screen-v0 Q-table size: {len(mc_screen.Q)} (vs v0: {len(mc_v0.Q)})")
    env_screen.close()
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['generalization', 'screen', 'all'])
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = {}
    
    if args.experiment in ['generalization', 'all']:
        gen_results = run_generalization_experiment(args.num_episodes, args.seed)
        all_results['generalization'] = gen_results
    
    if args.experiment in ['screen', 'all']:
        screen_results = run_screen_experiment(args.num_episodes, args.seed)
        all_results['screen_comparison'] = screen_results
    
    # Save — use experiment-specific filename to avoid overwriting other results
    out_path = os.path.join(args.output_dir,
                            f'{args.experiment}_results_seed{args.seed}.json')
    
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(out_path, 'w') as f:
        json.dump(all_results, f, default=convert)
    
    print(f"\n✅ All extra experiment results saved to {out_path}")


if __name__ == '__main__':
    main()
