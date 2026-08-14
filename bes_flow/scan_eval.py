# bes_flow/scan_eval.py
#
# Evaluate every checkpoint produced by hyperparam_scan_array.sh on the
# held-out test set, and plot a smooth_weight x laplacian_weight EPE
# heatmap for each model.
#
# Usage
# -----
#   python -m bes_flow.scan_eval

import os
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from dataclasses import replace

from bes_flow.config import cfg
from bes_flow.model_s import BESFlowNetS
from bes_flow.model_pwcnet import PWCNet
from bes_flow.dataset import make_datasets
from bes_flow.train import load_model, run_evaluation, resolve_cache_path


# ── Hyperparameter grid — keep in sync with hyperparam_scan_array.sh ───────
MODELS         = ['flownet', 'pwc']   
SMOOTH_WEIGHTS = [0.0, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032]
LAP_WEIGHTS    = [0.0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08]
CKPT_ROOT      = os.path.expandvars("$SCRATCH/bes_flow/checkpoints/scan")
OUT_DIR        = os.path.expandvars("$SCRATCH/bes_flow/outputs/scan")
# ─────────────────────────────────────────────────────────────────────────


def build_test_dataset(cfg):
    """
    Reproduce the exact frame split + dataset cache used by train.py, so
    this is the identical test set each model saw during its own
    evaluation. Only smooth_weight/laplacian_weight vary across the scan,
    not flow_type or max_shift, so every combo shares one cache file —
    this is loaded once, here, rather than once per checkpoint.
    """
    print(f"Loading BES frames: {cfg.data_path}")
    with h5py.File(cfg.data_path, 'r') as hf:
        all_frames = hf['images'][:]

    N       = len(all_frames)
    n_test  = int(cfg.test_split * N)
    n_val   = int(cfg.val_split * N)
    n_train = N - n_test - n_val

    rng     = np.random.default_rng(cfg.test_seed)
    indices = rng.permutation(N)
    train_frames = all_frames[indices[:n_train]]
    val_frames   = all_frames[indices[n_train:n_train + n_val]]
    test_frames  = all_frames[indices[n_train + n_val:]]

    cfg = replace(
        cfg,
        dataset_cache_path=resolve_cache_path(cfg.dataset_cache_path, cfg.flow_type),
    )

    print("Building datasets (cache hit expected — scan jobs already generated it)...")
    _, _, test_dataset = make_datasets(train_frames, val_frames, test_frames, cfg)
    return test_dataset, test_frames, cfg


def plot_heatmaps(epe_grid, out_dir):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(6 * len(MODELS), 5))
    if len(MODELS) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, MODELS):
        grid = epe_grid[model_name]
        im = ax.imshow(grid, cmap='viridis_r', aspect='auto', origin='lower')

        ax.set_xticks(range(len(LAP_WEIGHTS)))
        ax.set_xticklabels(LAP_WEIGHTS)
        ax.set_yticks(range(len(SMOOTH_WEIGHTS)))
        ax.set_yticklabels(SMOOTH_WEIGHTS)
        ax.set_xlabel('laplacian_weight')
        ax.set_ylabel('smooth_weight')
        ax.set_title(model_name)

        vmin, vmax = np.nanmin(grid), np.nanmax(grid)
        midpoint = (vmin + vmax) / 2 if np.isfinite(vmin) and np.isfinite(vmax) else 0
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                val = grid[i, j]
                text = f"{val:.3f}" if not np.isnan(val) else "—"
                color = 'white' if (not np.isnan(val) and val > midpoint) else 'black'
                ax.text(j, i, text, ha='center', va='center', color=color, fontsize=11)

        plt.colorbar(im, ax=ax, label='Mean test EPE (px)')

    fig.suptitle('Hyperparameter scan: test-set EPE', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'epe_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nSaved heatmap: {path}")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    test_dataset, test_frames, cfg_local = build_test_dataset(cfg)
    print(f"Test set: {len(test_dataset)} pairs\n")

    os.makedirs(OUT_DIR, exist_ok=True)

    model_builders = {
        'flownet': lambda: BESFlowNetS(),
        'pwc':  lambda: PWCNet(),
    }

    # epe_grid[model][i, j] = mean test EPE for (SMOOTH_WEIGHTS[i], LAP_WEIGHTS[j])
    epe_grid = {m: np.full((len(SMOOTH_WEIGHTS), len(LAP_WEIGHTS)), np.nan)
                for m in MODELS}

    for model_name in MODELS:
        for i, sw in enumerate(SMOOTH_WEIGHTS):
            for j, lw in enumerate(LAP_WEIGHTS):
                run_name  = f"{model_name}_sw{sw}_lw{lw}"
                ckpt_path = os.path.join(
                    CKPT_ROOT, run_name, f'model_{cfg_local.flow_type}_best.pt'
                )

                if not os.path.exists(ckpt_path):
                    print(f"[skip] {run_name}: no checkpoint at {ckpt_path}")
                    continue

                print(f"\n── {run_name} ──")
                model = model_builders[model_name]().to(device)
                model = load_model(model, ckpt_path, device, cfg_local)

                results = run_evaluation(
                    model,
                    test_dataset = test_dataset,
                    test_frames  = test_frames,
                    device       = device,
                    cfg          = cfg_local,
                    output_dir   = os.path.join(OUT_DIR, run_name),
                    plot_results = False,  # skip per-run figures, only need the scalar EPE
                )
                epe_grid[model_name][i, j] = results['EPE'].mean()

                del model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

    # Save the raw grid so the heatmap can be replotted without rerunning inference
    np.savez(
        os.path.join(OUT_DIR, 'epe_grid.npz'),
        smooth_weights=np.array(SMOOTH_WEIGHTS),
        lap_weights=np.array(LAP_WEIGHTS),
        **{model_name: epe_grid[model_name] for model_name in MODELS},
    )

    plot_heatmaps(epe_grid, OUT_DIR)