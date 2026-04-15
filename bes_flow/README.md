# bes_flow

Neural network optical flow for Beam Emission Spectroscopy (BES) plasma diagnostics.

## Project structure

```
── bes_flow/
    ├── config.py        ← ALL hyperparameters in one place
    ├── model_s.py       ← FlowNetS
    ├── loss.py          ← warping L2 loss + smoothness regulariser
    ├── dataset.py       ← synthetic pair generation + DataLoader
    └── train.py         ← training loop, validation, plotting
```

## Adjusting hyperparameters

Edit `bes_flow/config.py`. The most important settings are:

| Parameter          | Default | Effect                                  |
|--------------------|---------|-----------------------------------------|
| `max_shift`        | 6.0     | Maximum expected displacement in pixels |
| `max_displacement` | 4       | Correlation search radius (feature space)|
| `smooth_weight`    | 0.05    | Smoothness regularisation strength      |
| `num_epochs`       | 100     | Training duration                       |
| `batch_size`       | 16      | Frames per gradient step                |

## Architecture summary
See `model_s.py` and `model_pwcnet.py` for a detailed explanation of each component.

## References

- Quenot et al. (1998) — ODP baseline
- Dosovitskiy et al., ICCV 2015 — FlowNet
- Sun et al., CVPR 2018 — PWC-Net
- Meister et al., AAAI 2018 — UnFlow, unsupervised warping loss
