# bes_flow/train_jhopkins.py
#
# Optical flow model training on jhopkins turbulence dataset
#

import os
import json
import argparse
import numpy as np
import torch
from dataclasses import replace
 
from bes_flow.config  import cfg
from bes_flow.model_s import BESFlowNetS
from bes_flow.model_pwcnet import PWCNet
from bes_flow.loss    import WarpingL2Loss
from bes_flow.dataset import load_dataset_cache, make_dataloaders, BESDataset
from bes_flow.metrics import compute_all_metrics, print_summary
from bes_flow.predict import load_model
from bes_flow.train   import train, predict_dataset, plot_loss_history
 
 
if __name__ == '__main__':
 
    parser = argparse.ArgumentParser(
        description='Train a BES optical flow model on a JHTDB dataset'
    )
    parser.add_argument('--model', choices=['pwc', 'flownet'], default='pwc',
                        help='Model architecture: pwc (default) or flownet')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--checkpoint', default=None, metavar='FILE',
                        help='Resume from or fine-tune this checkpoint (.pt file)')
    parser.add_argument('--skip_train', action='store_true',
                        help='Skip training and go straight to evaluation')
    parser.add_argument('--plot_results', action='store_true',
                        help='Generate evaluation figures after training')
    args = parser.parse_args()
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}\n")
 
    # ── Override cfg fields relevant to this run ──────────────────────────
    # flow_type='jhopkins' drives checkpoint and history file names only
    run_cfg = replace(
        cfg,
        learning_rate  = 5e-5,
        num_epochs     = args.epochs,
        flow_type      = 'jhopkins',
        dataset_cache_path = os.path.expandvars(f"$SCRATCH/bes_flow/synthetic_data/dataset_jhopkins.h5")
    )
 
    # ── Load dataset cache ────────────────────────────────────────────────
    print(f"Loading dataset: {run_cfg.dataset_cache_path}")
    (train_A, train_B, train_flows,
     val_A,   val_B,   val_flows,
     test_A,  test_B,  test_flows,
     metadata) = load_dataset_cache(run_cfg.dataset_cache_path)
 
    print(f"  Train : {len(train_A)} pairs")
    print(f"  Val   : {len(val_A)} pairs")
    print(f"  Test  : {len(test_A)} pairs\n")
 
    train_dataset = BESDataset(train_A, train_B, train_flows, augment=True)
    val_dataset   = BESDataset(val_A,   val_B,   val_flows,   augment=False)
    test_dataset  = BESDataset(test_A,  test_B,  test_flows,  augment=False)
 
    train_loader, val_loader, test_loader = make_dataloaders(
        train_dataset, val_dataset, test_dataset, run_cfg
    )
 
    # ── Model ─────────────────────────────────────────────────────────────
    if args.model == 'flownet':
        print('Initializing BESFlowNetS')
        model = BESFlowNetS()
    else:
        print('Initializing PWCNet')
        model = PWCNet(max_displacement=run_cfg.max_displacement)
 
    model = model.to(device)
 
    if args.checkpoint is not None:
        model = load_model(model, args.checkpoint, device, run_cfg)
    elif args.model == 'flownet':
        args.checkpoint = os.path.expandvars(f"$SCRATCH/bes_flow/checkpoints-flownet/model_stage4_zonal_final.pt")
        model = load_model(model, args.checkpoint, device, run_cfg)
    elif args.model == 'pwc':
        args.checkpoint = os.path.expandvars(f"$SCRATCH/bes_flow/checkpoints-pwcnet/model_stage4_zonal_final.pt")
        model = load_model(model, args.checkpoint, device, run_cfg)
    #else:
    #    n_params = sum(p.numel() for p in model.parameters())
    #    print(f"Model parameters: {n_params:,}\n")
 
    # ── Loss ──────────────────────────────────────────────────────────────
    if run_cfg.is_supervised:
        print('\nRunning SUPERVISED training')
    else:
        print('\nRunning UNSUPERVISED training')
 
    loss_fn = WarpingL2Loss(
        smooth_weight    = run_cfg.smooth_weight,
        laplacian_weight = run_cfg.laplacian_weight,
        sup_weight       = run_cfg.sup_weight,
        is_supervised    = run_cfg.is_supervised,
    )
 
    # ── Train ─────────────────────────────────────────────────────────────
    if not args.skip_train:
        optimizer = torch.optim.Adam(model.parameters(), lr=run_cfg.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=run_cfg.num_epochs
        )
 
        print("Starting training...\n")
        loss_history = train(
            model, train_loader, val_loader,
            loss_fn, optimizer, scheduler,
            run_cfg, device,
        )
 
        history_path = run_cfg.output_dir + f'train_history_{run_cfg.flow_type}.json'
        with open(history_path, 'w') as f:
            json.dump(loss_history, f, indent=2)
        print(f"Loss history saved: {history_path}")
 
        if args.plot_results:
            plot_loss_history(loss_history, run_cfg)
 
    # ── Evaluate on the test set ──────────────────────────────────────────
    # Load the best checkpoint saved during training, or the one provided
    if args.checkpoint is not None and args.skip_train:
        best_ckpt = args.checkpoint
    else:
        best_ckpt = os.path.expandvars(f"$SCRATCH/bes_flow/checkpoints/model_{run_cfg.flow_type}_best.pt")
 
    print(f"\nLoading best checkpoint for evaluation: {best_ckpt}")
    model = load_model(model, best_ckpt, device, run_cfg)
 
    print(f"\n{'═' * 60}")
    print(f"  Evaluation on test set  ({len(test_dataset)} pairs)")
    print(f"{'═' * 60}\n")
 
    flows_pred = predict_dataset(model, test_dataset, device)
    flows_gt   = test_dataset.flows_gt
 
    results = compute_all_metrics(flows_pred, flows_gt)
    print_summary(results, flow_type='jhopkins', max_shift=None)
 
    print("Done.")
