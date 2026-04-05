"""
Generate all figures for the report.

Produces:
  results/figures/fig1_learning_curves.pdf
  results/figures/fig2_mc_value_policy.pdf
  results/figures/fig3_sarsa_value_policy.pdf
  results/figures/fig4_sweep_mc.pdf
  results/figures/fig5_sweep_sarsa.pdf
  results/figures/fig6_generalization.pdf
"""

import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns
from collections import defaultdict

from src.agents.mc_agent import MCControlAgent
from src.agents.sarsa_lambda import TrueOnlineSarsaLambda

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})
sns.set_style("whitegrid")

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
FIGS    = os.path.join(RESULTS, 'figures')
os.makedirs(FIGS, exist_ok=True)

MC_COLOR    = '#2196F3'
SARSA_COLOR = '#FF5722'


# ══════════════════════════════════════════════════════════════════════
# FIG 1 — Learning curves (3 seeds, mean ± std)
# ══════════════════════════════════════════════════════════════════════
def fig1_learning_curves():
    print("Generating fig1_learning_curves...")
    mc_runs, sarsa_runs = [], []
    for seed in [0, 1, 2]:
        path = os.path.join(RESULTS, f'training_results_h15_w20_g4_seed{seed}.json')
        with open(path) as f:
            d = json.load(f)
        mc_runs.append(d['mc']['rewards'])
        sarsa_runs.append(d['sarsa']['rewards'])

    def smooth_runs(runs, w=500):
        out = []
        for r in runs:
            k = np.ones(w) / w
            out.append(np.convolve(r, k, mode='valid'))
        return out

    def plot_with_ci(ax, runs, color, label, w=500):
        smoothed = smooth_runs(runs, w)
        n = min(len(s) for s in smoothed)
        arr = np.array([s[:n] for s in smoothed])
        mean, std = arr.mean(0), arr.std(0)
        ep = np.arange(n) + w - 1
        ax.plot(ep, mean, color=color, lw=2, label=label)
        ax.fill_between(ep, mean - std, mean + std, color=color, alpha=0.2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: both on same axis
    ax = axes[0]
    plot_with_ci(ax, mc_runs,    MC_COLOR,    'MC Control')
    plot_with_ci(ax, sarsa_runs, SARSA_COLOR, 'Sarsa(λ)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward (500-ep avg)')
    ax.set_title('Learning Curves — Training Reward')
    ax.legend()

    # Right: zoom on MC only (Sarsa scale dwarfs MC)
    ax = axes[1]
    plot_with_ci(ax, mc_runs, MC_COLOR, 'MC Control')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward (500-ep avg)')
    ax.set_title('MC Control — Convergence Detail')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, 'fig1_learning_curves.pdf'))
    plt.savefig(os.path.join(FIGS, 'fig1_learning_curves.png'))
    plt.close()
    print("  → fig1 done")


# ══════════════════════════════════════════════════════════════════════
# FIG 2 — MC value function + policy (rebuild agent from saved Q-table)
# ══════════════════════════════════════════════════════════════════════
def fig2_mc_value_policy():
    print("Generating fig2_mc_value_policy...")
    qtable_path = os.path.join(RESULTS, 'models', 'mc_q_table_h15_w20_g4_seed0.json')
    with open(qtable_path) as f:
        raw = json.load(f)

    # Reconstruct Q from saved dict
    Q = defaultdict(lambda: np.zeros(2))
    for k, v in raw.items():
        state = tuple(int(x) for x in k.strip('()').split(', '))
        Q[state] = np.array(v)

    # Get all observed (Δx, Δy) pairs
    x_vals = sorted(set(s[0] for s in Q))
    y_vals = sorted(set(s[1] for s in Q))

    # Build 2D arrays
    V      = np.full((len(x_vals), len(y_vals)), np.nan)
    policy = np.full((len(x_vals), len(y_vals)), np.nan)
    Q0     = np.full((len(x_vals), len(y_vals)), np.nan)
    Q1     = np.full((len(x_vals), len(y_vals)), np.nan)

    xi = {x: i for i, x in enumerate(x_vals)}
    yi = {y: i for i, y in enumerate(y_vals)}

    for state, q in Q.items():
        i, j = xi[state[0]], yi[state[1]]
        V[i, j]      = np.max(q)
        policy[i, j] = np.argmax(q)
        Q0[i, j], Q1[i, j] = q[0], q[1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Value function
    ax = axes[0]
    x_arr = np.array(x_vals)
    y_arr = np.array(y_vals)
    # Use pcolormesh for exact integer grid
    im = ax.pcolormesh(x_arr, y_arr, V.T, cmap='RdYlGn', shading='nearest')
    plt.colorbar(im, ax=ax, label='V(s) = max_a Q(s,a)')
    ax.set_xlabel('Δx (horizontal dist. to pipe)')
    ax.set_ylabel('Δy (vertical dist. to gap center)')
    ax.set_title('MC Control — State-Value Function V(s)')

    # Policy
    ax = axes[1]
    cmap2 = mcolors.ListedColormap(['#42A5F5', '#EF5350'])
    im2 = ax.pcolormesh(x_arr, y_arr, policy.T, cmap=cmap2,
                         vmin=-0.5, vmax=1.5, shading='nearest')
    patches = [mpatches.Patch(color='#42A5F5', label='No Flap (0)'),
               mpatches.Patch(color='#EF5350', label='Flap (1)')]
    ax.legend(handles=patches, loc='upper right')
    ax.set_xlabel('Δx (horizontal dist. to pipe)')
    ax.set_ylabel('Δy (vertical dist. to gap center)')
    ax.set_title('MC Control — Greedy Policy π(s)')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, 'fig2_mc_value_policy.pdf'))
    plt.savefig(os.path.join(FIGS, 'fig2_mc_value_policy.png'))
    plt.close()
    print("  → fig2 done")


# ══════════════════════════════════════════════════════════════════════
# FIG 3 — Sarsa value function + policy (from saved weights)
# ══════════════════════════════════════════════════════════════════════
def fig3_sarsa_value_policy():
    print("Generating fig3_sarsa_value_policy...")
    w_path = os.path.join(RESULTS, 'models', 'sarsa_weights_h15_w20_g4_seed0.npz')
    data = np.load(w_path)
    w = data['w']

    agent = TrueOnlineSarsaLambda(
        alpha=0.025, gamma=0.99, lam=0.9,
        num_tilings=8, num_tiles=8, iht_size=4096
    )
    agent.w = w
    agent.epsilon = 0.0

    x_vals = np.linspace(0, 19, 60)    # Δx range for h=15, w=20
    y_vals = np.linspace(-14, 14, 60)  # Δy range

    V      = np.zeros((len(x_vals), len(y_vals)))
    policy = np.zeros((len(x_vals), len(y_vals)))

    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            obs = np.array([x, y], dtype=float)
            qs = [agent._get_q(obs, a) for a in range(2)]
            V[i, j]      = max(qs)
            policy[i, j] = np.argmax(qs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    im = ax.pcolormesh(x_vals, y_vals, V.T, cmap='RdYlGn', shading='auto')
    plt.colorbar(im, ax=ax, label='V(s) = max_a q̂(s,a,w)')
    ax.set_xlabel('Δx (horizontal dist. to pipe)')
    ax.set_ylabel('Δy (vertical dist. to gap center)')
    ax.set_title('Sarsa(λ) — State-Value Function V(s)')
    ax.axhline(0, color='white', lw=0.8, ls='--', alpha=0.6)

    ax = axes[1]
    cmap2 = mcolors.ListedColormap(['#42A5F5', '#EF5350'])
    ax.pcolormesh(x_vals, y_vals, policy.T, cmap=cmap2,
                  vmin=-0.5, vmax=1.5, shading='auto')
    patches = [mpatches.Patch(color='#42A5F5', label='No Flap (0)'),
               mpatches.Patch(color='#EF5350', label='Flap (1)')]
    ax.legend(handles=patches, loc='upper right')
    ax.set_xlabel('Δx (horizontal dist. to pipe)')
    ax.set_ylabel('Δy (vertical dist. to gap center)')
    ax.set_title('Sarsa(λ) — Greedy Policy π(s)')
    ax.axhline(0, color='white', lw=0.8, ls='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, 'fig3_sarsa_value_policy.pdf'))
    plt.savefig(os.path.join(FIGS, 'fig3_sarsa_value_policy.png'))
    plt.close()
    print("  → fig3 done")


# ══════════════════════════════════════════════════════════════════════
# FIG 4 — MC sweep heatmap (alpha × gamma)
# ══════════════════════════════════════════════════════════════════════
def fig4_sweep_mc():
    print("Generating fig4_sweep_mc...")
    import itertools

    alphas = [0.01, 0.05, 0.1, 0.2, 0.5]
    gammas = [0.9, 0.95, 0.99, 1.0]

    grid = np.full((len(alphas), len(gammas)), np.nan)
    std_grid = np.full((len(alphas), len(gammas)), np.nan)

    for idx, (alpha, gamma) in enumerate(itertools.product(alphas, gammas)):
        path = os.path.join(RESULTS, f'sweep_mc_config{idx}.json')
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        i = alphas.index(alpha)
        j = gammas.index(gamma)
        grid[i, j]     = d['mean_eval']
        std_grid[i, j] = d['std_eval']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    sns.heatmap(grid, xticklabels=[str(g) for g in gammas],
                yticklabels=[str(a) for a in alphas],
                annot=True, fmt='.0f', cmap='YlOrRd', ax=ax,
                linewidths=0.5, cbar_kws={'label': 'Mean Eval Score'})
    ax.set_xlabel('γ (discount factor)')
    ax.set_ylabel('α (learning rate)')
    ax.set_title('MC Control — Mean Eval Score (↑ better)')
    # Mark best
    best = np.nanargmax(grid)
    bi, bj = np.unravel_index(best, grid.shape)
    ax.add_patch(plt.Rectangle((bj, bi), 1, 1, fill=False, edgecolor='lime', lw=3))

    ax = axes[1]
    sns.heatmap(std_grid, xticklabels=[str(g) for g in gammas],
                yticklabels=[str(a) for a in alphas],
                annot=True, fmt='.0f', cmap='Blues_r', ax=ax,
                linewidths=0.5, cbar_kws={'label': 'Std Eval Score'})
    ax.set_xlabel('γ (discount factor)')
    ax.set_ylabel('α (learning rate)')
    ax.set_title('MC Control — Eval Std (↓ more consistent)')
    ax.add_patch(plt.Rectangle((bj, bi), 1, 1, fill=False, edgecolor='lime', lw=3))

    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, 'fig4_sweep_mc.pdf'))
    plt.savefig(os.path.join(FIGS, 'fig4_sweep_mc.png'))
    plt.close()
    print("  → fig4 done")


# ══════════════════════════════════════════════════════════════════════
# FIG 5 — Sarsa sweep heatmap (alpha_base × lambda)
# ══════════════════════════════════════════════════════════════════════
def fig5_sweep_sarsa():
    print("Generating fig5_sweep_sarsa...")
    import itertools

    alpha_bases = [0.1, 0.2, 0.4, 0.8]
    lams        = [0.0, 0.4, 0.8, 0.9, 0.95, 1.0]

    grid     = np.full((len(alpha_bases), len(lams)), np.nan)
    std_grid = np.full((len(alpha_bases), len(lams)), np.nan)

    for idx, (ab, lam) in enumerate(itertools.product(alpha_bases, lams)):
        path = os.path.join(RESULTS, f'sweep_sarsa_config{idx}.json')
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        i = alpha_bases.index(ab)
        j = lams.index(lam)
        grid[i, j]     = d['mean_eval']
        std_grid[i, j] = d['std_eval']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    sns.heatmap(grid, xticklabels=[str(l) for l in lams],
                yticklabels=[str(a) for a in alpha_bases],
                annot=True, fmt='.0f', cmap='YlOrRd', ax=ax,
                linewidths=0.5, cbar_kws={'label': 'Mean Eval Score'})
    ax.set_xlabel('λ (trace decay)')
    ax.set_ylabel('α_base (learning rate × 8)')
    ax.set_title('Sarsa(λ) — Mean Eval Score (↑ better)')
    best = np.nanargmax(grid)
    bi, bj = np.unravel_index(best, grid.shape)
    ax.add_patch(plt.Rectangle((bj, bi), 1, 1, fill=False, edgecolor='lime', lw=3))

    ax = axes[1]
    sns.heatmap(std_grid, xticklabels=[str(l) for l in lams],
                yticklabels=[str(a) for a in alpha_bases],
                annot=True, fmt='.0f', cmap='Blues_r', ax=ax,
                linewidths=0.5, cbar_kws={'label': 'Std Eval Score'})
    ax.set_xlabel('λ (trace decay)')
    ax.set_ylabel('α_base (learning rate × 8)')
    ax.set_title('Sarsa(λ) — Eval Std (↓ more consistent)')
    ax.add_patch(plt.Rectangle((bj, bi), 1, 1, fill=False, edgecolor='lime', lw=3))

    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, 'fig5_sweep_sarsa.pdf'))
    plt.savefig(os.path.join(FIGS, 'fig5_sweep_sarsa.png'))
    plt.close()
    print("  → fig5 done")


# ══════════════════════════════════════════════════════════════════════
# FIG 6 — Generalization bar chart
# ══════════════════════════════════════════════════════════════════════
def fig6_generalization():
    print("Generating fig6_generalization...")

    configs = [
        'Baseline\ng=4',
        'Harder\ng=3',
        'Easier\ng=6',
        'Bigger\n20×30',
        'Smaller\n10×15',
    ]
    mc_means    = [240.3, 21.1, 747.4, 4.0, 105.3]
    mc_stds     = [205.0, 18.3, 686.4, 0.0, 104.3]
    sarsa_means = [201.8, 21.6, 645.4, 4.0,  61.9]
    sarsa_stds  = [200.0, 13.7, 588.9, 0.0,  63.3]

    x = np.arange(len(configs))
    w = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    bars1 = ax.bar(x - w/2, mc_means,    w, yerr=mc_stds,    capsize=4,
                   color=MC_COLOR,    label='MC Control',   alpha=0.85)
    bars2 = ax.bar(x + w/2, sarsa_means, w, yerr=sarsa_stds, capsize=4,
                   color=SARSA_COLOR, label='Sarsa(λ)',      alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=10)
    ax.set_ylabel('Mean Eval Score (100 episodes, cap=3000)')
    ax.set_title('Generalization: Trained on Default (H=15, W=20, g=4), Tested on Variants')
    ax.legend()
    ax.set_ylim(0, 1050)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, 'fig6_generalization.pdf'))
    plt.savefig(os.path.join(FIGS, 'fig6_generalization.png'))
    plt.close()
    print("  → fig6 done")


# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f"Saving figures to {FIGS}\n")
    fig1_learning_curves()
    fig2_mc_value_policy()
    fig3_sarsa_value_policy()
    fig4_sweep_mc()
    fig5_sweep_sarsa()
    fig6_generalization()
    print("\nAll figures generated.")
