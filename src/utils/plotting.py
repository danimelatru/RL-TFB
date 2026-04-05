"""
Plotting utilities for RL Assignment.

Provides functions for:
- State/Action-value function heatmaps
- Learning curves with smoothing
- Parameter sweep visualizations
- Policy visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from typing import Dict, List, Tuple, Optional


# ── Style ──────────────────────────────────────────────────────────────
def setup_plot_style():
    """Configure matplotlib for publication-quality plots."""
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.figsize': (10, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })
    sns.set_palette("deep")


# ── Learning Curves ────────────────────────────────────────────────────
def smooth(data: List[float], window: int = 100) -> np.ndarray:
    """Apply rolling average smoothing."""
    if len(data) < window:
        return np.array(data)
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def plot_learning_curves(agents_data: Dict[str, Dict], 
                         metric: str = 'reward',
                         window: int = 100,
                         title: str = None,
                         ylabel: str = None,
                         save_path: str = None):
    """Plot learning curves for multiple agents.
    
    Args:
        agents_data: Dict mapping agent_name → {'rewards': [...], 'lengths': [...]}
        metric: 'reward' or 'length'
        window: Smoothing window size
        title: Plot title
        ylabel: Y-axis label
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']
    
    for i, (name, data) in enumerate(agents_data.items()):
        key = 'rewards' if metric == 'reward' else 'lengths'
        values = data[key]
        smoothed = smooth(values, window)
        
        # Plot raw (transparent) + smoothed
        episodes = np.arange(len(values))
        ax.plot(episodes, values, alpha=0.1, color=colors[i % len(colors)])
        smoothed_episodes = np.arange(window - 1, len(values))
        ax.plot(smoothed_episodes, smoothed, 
                color=colors[i % len(colors)], linewidth=2, label=name)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel(ylabel or metric.capitalize())
    ax.set_title(title or f'Learning Curve ({metric})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_learning_curves_with_ci(runs_data: Dict[str, List[List[float]]],
                                  window: int = 100,
                                  title: str = 'Learning Curves',
                                  ylabel: str = 'Episode Reward',
                                  save_path: str = None):
    """Plot learning curves with confidence intervals from multiple seeds.
    
    Args:
        runs_data: Dict mapping agent_name → list of reward lists (one per seed)
        window: Smoothing window
        title: Plot title
        ylabel: Y-axis label
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']
    
    for i, (name, runs) in enumerate(runs_data.items()):
        # Smooth each run
        smoothed_runs = [smooth(run, window) for run in runs]
        min_len = min(len(s) for s in smoothed_runs)
        smoothed_runs = [s[:min_len] for s in smoothed_runs]
        
        runs_array = np.array(smoothed_runs)
        mean = runs_array.mean(axis=0)
        std = runs_array.std(axis=0)
        
        episodes = np.arange(min_len) + window - 1
        
        ax.plot(episodes, mean, color=colors[i % len(colors)], 
                linewidth=2, label=name)
        ax.fill_between(episodes, mean - std, mean + std, 
                        color=colors[i % len(colors)], alpha=0.2)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ── Value Function Plots ───────────────────────────────────────────────
def plot_value_function_grid(V_grid: np.ndarray,
                              x_labels: np.ndarray = None,
                              y_labels: np.ndarray = None,
                              title: str = 'State-Value Function V(s)',
                              save_path: str = None):
    """Plot value function as a heatmap.
    
    Args:
        V_grid: 2D array (x_bins × y_bins) of state values
        x_labels: Tick labels for x-axis (Δx bins)
        y_labels: Tick labels for y-axis (Δy bins)
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Transpose so x is horizontal, y is vertical
    im = ax.imshow(V_grid.T, cmap='RdBu_r', aspect='auto', origin='lower',
                   interpolation='nearest')
    
    ax.set_xlabel('Δx (Horizontal Distance to Pipe)')
    ax.set_ylabel('Δy (Vertical Distance to Gap Center)')
    ax.set_title(title)
    
    if x_labels is not None:
        step = max(1, len(x_labels) // 10)
        ax.set_xticks(np.arange(0, len(x_labels), step))
        ax.set_xticklabels([f'{x:.1f}' for x in x_labels[::step]], rotation=45)
    if y_labels is not None:
        step = max(1, len(y_labels) // 10)
        ax.set_yticks(np.arange(0, len(y_labels), step))
        ax.set_yticklabels([f'{y:.1f}' for y in y_labels[::step]])
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('V(s) = max_a Q(s,a)')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_policy_grid(policy_grid: np.ndarray,
                      x_labels: np.ndarray = None,
                      y_labels: np.ndarray = None,
                      title: str = 'Learned Policy π(s)',
                      action_names: List[str] = None,
                      save_path: str = None):
    """Plot policy as a heatmap (action per state).
    
    Args:
        policy_grid: 2D array (x_bins × y_bins) of action indices
        title: Plot title
        action_names: List of action names for legend
        save_path: Path to save
    """
    if action_names is None:
        action_names = ['No Flap (0)', 'Flap (1)']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cmap = mcolors.ListedColormap(['#42A5F5', '#EF5350'])
    bounds = [-0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    im = ax.imshow(policy_grid.T, cmap=cmap, norm=norm, aspect='auto',
                   origin='lower', interpolation='nearest')
    
    ax.set_xlabel('Δx (Horizontal Distance to Pipe)')
    ax.set_ylabel('Δy (Vertical Distance to Gap Center)')
    ax.set_title(title)
    
    if x_labels is not None:
        step = max(1, len(x_labels) // 10)
        ax.set_xticks(np.arange(0, len(x_labels), step))
        ax.set_xticklabels([f'{x:.1f}' for x in x_labels[::step]], rotation=45)
    if y_labels is not None:
        step = max(1, len(y_labels) // 10)
        ax.set_yticks(np.arange(0, len(y_labels), step))
        ax.set_yticklabels([f'{y:.1f}' for y in y_labels[::step]])
    
    # Create legend patches
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=cmap(0), label=action_names[0]),
               mpatches.Patch(color=cmap(1), label=action_names[1])]
    ax.legend(handles=patches, loc='upper right')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_value_and_policy(V_grid, policy_grid, x_labels=None, y_labels=None,
                           title_prefix='', save_path=None):
    """Side-by-side value function and policy plots."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Value function
    im1 = axes[0].imshow(V_grid.T, cmap='RdBu_r', aspect='auto', 
                         origin='lower', interpolation='nearest')
    axes[0].set_xlabel('Δx')
    axes[0].set_ylabel('Δy')
    axes[0].set_title(f'{title_prefix}State-Value Function V(s)')
    plt.colorbar(im1, ax=axes[0], shrink=0.8)
    
    # Policy
    cmap = mcolors.ListedColormap(['#42A5F5', '#EF5350'])
    bounds = [-0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    im2 = axes[1].imshow(policy_grid.T, cmap=cmap, norm=norm, aspect='auto',
                         origin='lower', interpolation='nearest')
    axes[1].set_xlabel('Δx')
    axes[1].set_ylabel('Δy')
    axes[1].set_title(f'{title_prefix}Policy π(s)')
    
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=cmap(0), label='No Flap'),
               mpatches.Patch(color=cmap(1), label='Flap')]
    axes[1].legend(handles=patches, loc='upper right')
    
    if x_labels is not None:
        for ax in axes:
            step = max(1, len(x_labels) // 8)
            ax.set_xticks(np.arange(0, len(x_labels), step))
            ax.set_xticklabels([f'{x:.0f}' for x in x_labels[::step]], rotation=45)
    if y_labels is not None:
        for ax in axes:
            step = max(1, len(y_labels) // 8)
            ax.set_yticks(np.arange(0, len(y_labels), step))
            ax.set_yticklabels([f'{y:.0f}' for y in y_labels[::step]])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ── Parameter Sweep ────────────────────────────────────────────────────
def plot_parameter_sweep_heatmap(results: np.ndarray,
                                  x_params: List[float],
                                  y_params: List[float],
                                  xlabel: str = 'α',
                                  ylabel: str = 'λ',
                                  title: str = 'Parameter Sweep',
                                  save_path: str = None):
    """Plot parameter sweep results as a heatmap.
    
    Args:
        results: 2D array of performance values (len(x_params), len(y_params))
        x_params, y_params: Parameter values
        xlabel, ylabel: Axis labels
        title: Plot title
        save_path: Path to save
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(results.T, xticklabels=[f'{p:.3f}' for p in x_params],
                yticklabels=[f'{p:.2f}' for p in y_params],
                annot=True, fmt='.1f', cmap='YlOrRd', ax=ax)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_comparison_bar(agent_results: Dict[str, float],
                         metric_name: str = 'Average Score',
                         title: str = 'Agent Comparison',
                         save_path: str = None):
    """Bar chart comparing agents on a metric.
    
    Args:
        agent_results: Dict mapping agent_name → metric_value
        metric_name: Name of the metric
        title: Plot title
        save_path: Path to save
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    names = list(agent_results.keys())
    values = list(agent_results.values())
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0'][:len(names)]
    
    bars = ax.bar(names, values, color=colors, edgecolor='white', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
