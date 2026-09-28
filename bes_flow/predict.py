# bes_flow/predict.py
#
# Run optical flow inference on an experimental BES HDF5 file using up to
# five methods, save per-method results and optionally plot a Vr radial profile.
#
# Methods:
#   1. PWCNet         (--weights_pwc)
#   2. BESFlowNetS    (--weights_flownets)
#   3. RAFT-small     (torchvision pretrained)
#   4. Farneback      (OpenCV)
#   5. ODP            (bes_flow.odp)
#
# Output files (one per method, alongside the input file):
#   <stem>_pwc.h5, <stem>_flownet.h5, <stem>_raft.h5,
#   <stem>_farneback.h5, <stem>_odp.h5
#
# Each output contains:
#   vR, vZ            - velocity arrays (m/s), shape (n_frames, 8, 8)
#   R, Z              - spatial coordinates (m) at 8x8 resolution
#   time              - time axis (ms) for the n_frames velocity frames
#   R_profile         - R coordinates for the radial profile (full 64-pt grid)
#   vZ_profile        - vZ averaged over time and Z (64-pt radial profile)
#
# Usage
# ─────
#   python predict.py \
#       --input  data/shot12345.h5 \
#       --weights_pwc       checkpoints/pwc_best.pt \
#       --weights_flownets  checkpoints/flownets_best.pt \
#       [--skip_raft] [--skip_farneback] [--skip_odp] \
#       [--skip_pwc] [--skip_flownets] \
#       [--plot] \
#       [--plot_quiver] [--quiver_frames 100 200 300 400] \
#       [--quiver_step 8] [--quiver_scale 2000]
#
#   # re-plot from previously saved results, no inference:
#   python predict.py --input data/shot12345.h5 --load_saved --plot --plot_quiver

import os
import argparse
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import torch

from bes_flow.compare_methods import (
    load_pwc,
    load_flownets,
    run_farneback,
    run_raft_small,
    run_odp,
)


def load_bes_h5(path):
    """
    Load images and coordinate axes from a BES HDF5 file.

    Expected entries: 'images', 'time', 'R', 'Z'
      images : (N, H, W)  or  (N, 1, H, W)
      time   : (N,)  [ms]
      R      : (W,)  radial positions  [cm]
      Z      : (H,)  poloidal positions [cm]

    Returns
    -------
    images : (N, H, W) float32
    time   : (N,) float32
    R      : (W,) float32
    Z      : (H,) float32
    """
    with h5py.File(path, 'r') as f:
        images = f['images'][()].astype(np.float32)
        time   = f['time'][()].astype(np.float32)
        R      = f['R'][()].astype(np.float32)
        Z      = f['Z'][()].astype(np.float32)

    if images.ndim == 4:          # (N, 1, H, W) -> (N, H, W)
        images = images[:, 0]

    print(f"  Loaded {images.shape[0]} frames  ({images.shape[1]}x{images.shape[2]} px)")
    print(f"  R: {R[0]:.2f} - {R[-1]:.2f} cm ({len(R)} pts)")
    print(f"  Z: {Z[0]:.2f} - {Z[-1]:.2f} cm ({len(Z)} pts)")
    return images, time, R, Z


def normalize_sequence(images):
    """
    Joint normalization across the whole sequence to [0, 1].
    Returns float32 array with the same shape as input.
    """
    vmin = images.min()
    vmax = images.max()
    if vmax > vmin:
        return (images - vmin) / (vmax - vmin)
    return images


def make_pairs(images, per_pair_norm=False):
    """
    Build consecutive frame pairs.
 
    Parameters
    ----------
    images        : (N, H, W) float32
    per_pair_norm : bool
        If True, each pair (A, B) is normalised jointly to [0, 1] using the
        min/max across both frames. 

    Returns
    -------
    framesA : (N-1, 1, H, W)
    framesB : (N-1, 1, H, W)
    """
    framesA = images[:-1, np.newaxis].copy()   # (N-1, 1, H, W)
    framesB = images[1:,  np.newaxis].copy()
 
    if per_pair_norm:
        # Per-pair min/max across both frames
        # Flatten spatial dims to (N, 2*H*W), then reduce along that axis.
        flat   = np.concatenate([framesA, framesB], axis=1).reshape(len(framesA), -1)
        vmin   = flat.min(axis=1)[:, None, None, None]   # (N, 1, 1, 1)
        vmax   = flat.max(axis=1)[:, None, None, None]
        scale  = np.where(vmax - vmin > 1e-6, vmax - vmin, 1.0)
        framesA = (framesA - vmin) / scale
        framesB = (framesB - vmin) / scale
 
    return framesA, framesB


# ─────────────────────────────────────────────────────────────────────────────
# Inference wrappers
# ─────────────────────────────────────────────────────────────────────────────
# Convention: all wrappers return (N, 2, H, W) float32
#   channel 0 = dx (R direction, pixels/frame)
#   channel 1 = dy (Z direction, pixels/frame)

def run_bes_model(model, framesA, framesB, device, batch_size=16,
                  per_frame_norm=True):
    """
    Generic runner for PWCNet and BESFlowNetS.
 
    Parameters
    ----------
    per_frame_norm : bool
        If True (default), each frame pair is normalised jointly to [0, 1]
    """
    N = len(framesA)
    H, W = framesA.shape[2], framesA.shape[3]
    flows = np.zeros((N, 2, H, W), dtype=np.float32)
 
    model.eval()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            bA  = torch.from_numpy(framesA[start:end]).to(device)
            bB  = torch.from_numpy(framesB[start:end]).to(device)
 
            if per_frame_norm:
                # Normalise each pair jointly: min/max over both frames
                pair     = torch.cat([bA, bB], dim=1)   # (B, 2, H, W)
                vmin     = pair.flatten(1).min(dim=1).values[:, None, None, None]
                vmax     = pair.flatten(1).max(dim=1).values[:, None, None, None]
                scale    = (vmax - vmin).clamp(min=1e-6)
                bA       = (bA - vmin) / scale
                bB       = (bB - vmin) / scale
 
            flows[start:end] = model(bA, bB).cpu().numpy()
 
    return flows


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 output
# ─────────────────────────────────────────────────────────────────────────────

# Method key -> default display label, used when reloading saved results
_METHOD_LABELS = {
    'pwc':       'PWC',
    'flownet':   'FlowNetS',
    'odp':       'ODP',
    'raft':      'RAFT-small',
    'farneback': 'Farneback',
}


def save_result(out_path, vR, vZ, R, Z, time_pairs, entry):
    """
    Write velocimetry results to an HDF5 file.

    Datasets
    --------
    vR, vZ                 : (n_frames, n_Z, n_R)  [m/s]
    R                      : (n_R,)  [cm]
    Z                      : (n_Z,)  [cm]
    time                   : (n_frames,)  [ms]
    R_profile              : (W,)   R grid for the profiles [cm]
    vR_profile, vZ_profile : (W,)   v(R) time-and-Z averaged [m/s]
    ReynoldsStress_profile : (W,)   <vR~ vZ~>(R)  [m^2/s^2]
    flux_profile           : (W,)   <vR~ n~>(R)   [a.u.]
    vR_frames, vZ_frames   : (n_sel, H, W) full-res velocities [m/s], only for
                             the frames selected for the quiver overlay
    quiver_frames          : (n_sel,) pair indices of those frames

    Attributes: label, method_key

    Everything needed by `plot_v_profile` and `plot_flow_quivers` is written
    here, so a later run with --load_saved can re-plot without re-running
    inference.
    """
    with h5py.File(out_path, 'w') as f:
        f.attrs['label']      = entry['label']
        f.attrs['method_key'] = entry['method_key']

        f.create_dataset('vR',   data=vR, compression='gzip')
        f.create_dataset('vZ',   data=vZ, compression='gzip')
        f.create_dataset('R',    data=R)
        f.create_dataset('Z',    data=Z)
        f.create_dataset('time', data=time_pairs)

        for key in ('R_profile', 'vR_profile', 'vZ_profile',
                    'ReynoldsStress_profile', 'flux_profile'):
            f.create_dataset(key, data=entry[key])

        if 'vR_frames' in entry:
            f.create_dataset('quiver_frames', data=entry['quiver_frames'])
            f.create_dataset('vR_frames', data=entry['vR_frames'], compression='gzip')
            f.create_dataset('vZ_frames', data=entry['vZ_frames'], compression='gzip')
    print(f"  Saved -> {out_path}")


def load_result(path, method_key=None):
    """
    Read back a file written by `save_result` into the dict format consumed by
    `plot_v_profile` / `plot_flow_quivers`. Returns None if the file predates
    the extended format and lacks the profile datasets.
    """
    with h5py.File(path, 'r') as f:
        key   = f.attrs.get('method_key', method_key or '')
        label = f.attrs.get('label', _METHOD_LABELS.get(key, key))
        if isinstance(key, bytes):
            key = key.decode()
        if isinstance(label, bytes):
            label = label.decode()

        missing = [k for k in ('R_profile', 'vR_profile', 'vZ_profile',
                               'ReynoldsStress_profile', 'flux_profile')
                   if k not in f]
        if missing:
            print(f"  [skip] {os.path.basename(path)} is missing {missing} "
                  f"— written by an older version, re-run inference for it")
            return None

        entry = {
            'label':      label,
            'method_key': key,
        }
        for k in ('R_profile', 'vR_profile', 'vZ_profile',
                  'ReynoldsStress_profile', 'flux_profile'):
            entry[k] = f[k][()]

        if 'vR_frames' in f:
            entry['quiver_frames'] = f['quiver_frames'][()]
            entry['vR_frames']     = f['vR_frames'][()]
            entry['vZ_frames']     = f['vZ_frames'][()]

    return entry


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

_METHOD_COLORS = {
    'pwc':      'steelblue',
    'flownet':  'darkorange',
    'odp':     'forestgreen',
    'farneback':'mediumpurple',
    'raft':      'crimson',
}

def plot_v_profile(results, velocity_component='Z', output_path=None):
    """
    Single figure with two side-by-side panels:
      left  — V radial profile (time and Z averaged)
      right — Reynolds stress <Vr*Vz> radial profile (time and Z averaged)
 
    Parameters
    ----------
    results              : list of dicts, each with keys
                             'label', 'R_profile', 'vR_profile', 'vZ_profile',
                             'ReynoldsStress_profile'
    velocity_component   : 'R' to plot Vr (default), 'Z' to plot Vz
    output_path          : str or None - save figure if given
    """
    if velocity_component == 'R':
        v_key   = 'vR_profile'
        v_label = r'$\langle V_R \rangle$  (m/s)'
        v_title = r'$\langle V_R \rangle$ - time & Z averaged'
    elif velocity_component == 'Z':
        v_key   = 'vZ_profile'
        v_label = r'$\langle V_Z \rangle$  (m/s)'
        v_title = r'$\langle V_Z \rangle$ - time & Z averaged'
    else:
        raise ValueError(f"velocity_component must be 'R' or 'Z', got {velocity_component!r}")
 
    fig, (ax_v, ax_rs, ax_fl) = plt.subplots(1, 3, figsize=(18, 4), sharex=True)
 
    for res in results:
        label = res['label']
        R     = res['R_profile']
        color = _METHOD_COLORS.get(res['method_key'], None)
        ax_v.plot(R,  res[v_key], label=label, color=color, lw=4)
        ax_rs.plot(R, res['ReynoldsStress_profile'],  label=label, color=color, lw=4)
        ax_fl.plot(R, res['flux_profile'],  label=label, color=color, lw=4)
 
    ax_v.set_xlabel('R  (cm)', fontsize=12)
    ax_v.set_ylabel(v_label, fontsize=12)
    ax_v.set_title(v_title, fontsize=14)
    ax_v.legend(fontsize=14)
    ax_v.grid(True, alpha=0.3)
    ax_v.axhline(0, color='k', lw=0.8, ls='--')
 
    ax_rs.set_xlabel('R  (cm)', fontsize=12)
    ax_rs.set_ylabel(r'$\langle \tilde{V}_R \tilde{V}_Z \rangle  (m^2/s^2)$', fontsize=12)
    ax_rs.set_title(r'Reynolds stress $\langle \tilde{V}_R \tilde{V}_Z \rangle$ - time & Z averaged', fontsize=13)
    ax_rs.legend(fontsize=14)
    ax_rs.grid(True, alpha=0.3)
    ax_rs.axhline(0, color='k', lw=0.8, ls='--')

    ax_fl.set_xlabel('R  (cm)', fontsize=12)
    #ax_fl.set_ylabel(r'$\langle \tilde{V}_R \tilde{n} \rangle  (m^{-2}s^{-1})$', fontsize=12)
    ax_fl.set_ylabel(r'$\langle \tilde{V}_R \tilde{n} \rangle  (a.u.)$', fontsize=12)
    ax_fl.set_title(r'Particle flux $\langle \tilde{V}_R \tilde{n} \rangle$ - time & Z averaged', fontsize=13)
    ax_fl.legend(fontsize=14)
    ax_fl.grid(True, alpha=0.3)
    ax_fl.axhline(0, color='k', lw=0.8, ls='--')
 
    fig.tight_layout()
 
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot -> {output_path}")
    plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Quiver overlay: frames x methods
# ─────────────────────────────────────────────────────────────────────────────

def _axis_limits(x):
    """
    Half-cell-padded limits for a 1-D coordinate array of cell centres,
    preserving the array's own ordering. Equivalent to the `extent` an
    `imshow(..., origin='lower')` call would use, so a pcolormesh panel lines
    up with (and points the same way as) the imshow version.
    Returns (lo, hi) which are reversed when the coordinate array descends.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        return float(x[0]) - 0.5, float(x[0]) + 0.5
    d = np.diff(x).mean()                      # signed
    return float(x[0] - d / 2), float(x[-1] + d / 2)


def plot_flow_quivers(results, images, R, Z, time_pairs, frame_indices,
                      quiver_step=8, scale=None, arrow_frac=0.9,
                      cmap='inferno', percentile=98.0, flip_z=False,
                      output_path=None):
    """
    Grid of BES frames with the estimated flow field overlaid as a quiver.

    Layout: one ROW per method, one COLUMN per requested frame.
    All panels share the same image color scale and the same arrow scale, so
    arrow lengths are directly comparable between methods and between frames.

    Parameters
    ----------
    results       : list of dicts as built by `postprocess_flows`; each must
                    carry 'label', 'method_key', 'vR_frames', 'vZ_frames'
                    with shape (n_sel, H, W) in m/s, matching `frame_indices`.
    images        : (N, H, W) raw (un-normalised) BES images
    R             : (W,) radial coordinates of the image grid [cm]
    Z             : (H,) poloidal coordinates of the image grid [cm]
    time_pairs    : (n_pairs,) time of frame A of each pair [ms]
    frame_indices : sequence of int, pair indices to display
    quiver_step   : int, arrows are sampled every `quiver_step` pixels starting
                    at an offset of quiver_step//2, i.e.
                    ys = arange(qs//2, H, qs), xs = arange(qs//2, W, qs).
                    With 64x64 interpolated images and qs=8 this lands one
                    arrow per native BES channel.
    scale         : matplotlib quiver `scale` (m/s per cm of arrow length).
                    None -> derived from the data so the largest arrows span
                    `arrow_frac` of one arrow cell.
    arrow_frac    : target arrow length as a fraction of the arrow cell size.
    percentile    : percentile of |v| used as the reference magnitude.
    flip_z        : force the Z axis to increase upward regardless of the
                    ordering of `Z` in the file. Default False, which keeps
                    the file's own ordering so the panels match
                    `imshow(frame, origin='lower',
                            extent=[R[0], R[-1], Z[0], Z[-1]])`.
    output_path   : str or None - save figure if given

    Notes
    -----
    `pcolormesh(R, Z, frame)` places image row j at coordinate Z[j], the same
    row->coordinate mapping as `imshow(origin='lower')` with the extent above,
    so no vertical flip of the data is needed. Only the axis *direction* is
    pinned explicitly (see `_axis_limits`), because pcolormesh would otherwise
    autoscale a descending Z axis into ascending order and mirror the panel
    relative to the imshow version. Arrow directions follow the axes: vZ > 0
    always points toward larger Z, whichever way that is on screen.
    """
    if not results:
        print("\nNo results to plot — all methods were skipped.")
        return
    if len(frame_indices) == 0:
        print("\nNo frames requested for the quiver plot.")
        return

    n_rows = len(results)
    n_cols = len(frame_indices)

    # ── Arrow sample points: every `quiver_step` pixels, half-step offset ────
    H, W   = np.asarray(results[0]['vR_frames']).shape[-2:]
    qs     = max(1, int(quiver_step))
    ys     = np.arange(qs // 2, H, qs)
    xs     = np.arange(qs // 2, W, qs)
    if len(ys) == 0:
        ys = np.array([H // 2])
    if len(xs) == 0:
        xs = np.array([W // 2])
    xx, yy = np.meshgrid(xs, ys)                       # (n_y, n_x) pixel idx

    R_a = np.asarray(R, dtype=np.float64)[xs]
    Z_a = np.asarray(Z, dtype=np.float64)[ys]
    RR, ZZ = np.meshgrid(R_a, Z_a)                     # (n_y, n_x) [cm]

    # ── Sample every method's selected frames at those points ───────────────
    sampled = []
    for res in results:
        vR_s = np.asarray(res['vR_frames'])[:, yy, xx]  # (n_sel, n_y, n_x)
        vZ_s = np.asarray(res['vZ_frames'])[:, yy, xx]
        sampled.append((vR_s, vZ_s))

    # ── Common arrow scale across all panels ────────────────────────────────
    if scale is None:
        mags = np.concatenate([np.hypot(a, b).ravel() for a, b in sampled])
        v_ref = np.percentile(mags, percentile)
        if not np.isfinite(v_ref) or v_ref <= 0:
            v_ref = 1.0
        cell = min(abs(np.diff(R_a)).mean() if len(R_a) > 1 else 1.0,
                   abs(np.diff(Z_a)).mean() if len(Z_a) > 1 else 1.0)
        scale = v_ref / (arrow_frac * cell)             # (m/s) per cm
    else:
        v_ref = scale * arrow_frac * (abs(np.diff(R_a)).mean() if len(R_a) > 1 else 1.0)

    # ── Common image color scale over the displayed frames ──────────────────
    imgs = images[list(frame_indices)]                  # (n_cols, H, W)
    im_vmin, im_vmax = np.percentile(imgs, [1.0, 99.0])

    # ── Axis limits (direction inherited from the coordinate arrays) ────────
    R_lim = _axis_limits(R)
    Z_lim = _axis_limits(Z)
    if flip_z:
        Z_lim = (min(Z_lim), max(Z_lim))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.4 * n_cols, 3.2 * n_rows),
                             squeeze=False, sharex=True, sharey=True)

    mesh = None
    for i, (res, (vR_s, vZ_s)) in enumerate(zip(results, sampled)):
        color = _METHOD_COLORS.get(res['method_key'], 'white')
        for j, idx in enumerate(frame_indices):
            ax = axes[i][j]
            mesh = ax.pcolormesh(R, Z, imgs[j], cmap=cmap,
                                 vmin=im_vmin, vmax=im_vmax, shading='auto')
            q = ax.quiver(RR, ZZ, vR_s[j], vZ_s[j],
                          angles='xy', scale_units='xy', scale=scale,
                          color=color, width=0.01, headwidth=4,)

            if i == 0:
                ax.set_title(f"t = {time_pairs[idx]:.3f} ms", fontsize=12)
            if j == 0:
                ax.set_ylabel(f"{res['label']}\nZ  (cm)", fontsize=11)
            if i == n_rows - 1:
                ax.set_xlabel('R  (cm)', fontsize=11)
            if j == n_cols - 1:
                # Reference arrow to the right of the last column, clear of the
                # column titles and of the shared colorbar.
                ax.quiverkey(q, 1.06, 0.5, v_ref, f"{v_ref:.0f} m/s",
                             labelpos='N', coordinates='axes',
                             fontproperties={'size': 9})

            # Pin the limits so the panel matches imshow(origin='lower',
            # extent=[R[0], R[-1], Z[0], Z[-1]]): half-cell padding, and the
            # axis direction taken from the coordinate arrays themselves.
            ax.set_xlim(*R_lim)
            ax.set_ylim(*Z_lim)
            ax.set_aspect('equal', adjustable='box')

    fig.suptitle('BES frames with overlaid flow field', fontsize=15)
    fig.tight_layout(rect=[0, 0, 0.89, 0.97])

    if mesh is not None:
        cax = fig.add_axes([0.945, 0.10, 0.012, 0.78])
        fig.colorbar(mesh, cax=cax, label='BES intensity (a.u.)')

    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot -> {output_path}")
    plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Flow postrpocessing
# ─────────────────────────────────────────────────────────────────────────────

def postprocess_flows(flows_px, frames, R_interp, Z_interp, time_pairs,
                      orig_res, results_to_plot, stem, method_key, label,
                      quiver_frames=None):
    """
    Convert pixel/frame flows to physical units, downsample to orig_res,
    compute profiles, and save results

    Parameters
    ----------
    flows_px  : (N, 2, H, W) float32  [pixels/frame]
                channel 0 = vR direction (x / R axis)
                channel 1 = vZ direction (y / Z axis)
    frames    : (N, H, W) float32, array of frameAs
    R_interp  : (W,) R coordinates of interpolated images [cm]
    Z_interp  : (H,) Z coordinates of interpolated images [cm]
    time_pairs: (N,) time of frame A in each pair [ms]
    orig_res  : (n_R, n_Z) output resolution, default (8, 8)
    quiver_frames : sequence of int or None
                Pair indices whose full-resolution velocity fields are kept in
                the result dict (as 'vR_frames'/'vZ_frames') for the quiver
                overlay plot. Only these frames are retained, so the memory
                cost stays negligible.

    Returns
    -------
    vR_down  : (N, n_Z, n_R) [m/s]
    vZ_down  : (N, n_Z, n_R) [m/s]
    R_down   : (n_R,) [cm]
    Z_down   : (n_Z,) [cm]
    vR_full  : (N, H, W) full-resolution vR in m/s  (for profiles)
    vZ_full  : (N, H, W) full-resolution vZ in m/s  (for Reynolds stress profile)
    """
    print(f"\nPost-processing {label}...")
    vR_full = flows_px[:, 0].copy()   # (N, H, W)
    vZ_full = flows_px[:, 1].copy()

    dR = (R_interp[1] - R_interp[0]) / 100.0   # cm -> m
    dZ = (Z_interp[1] - Z_interp[0]) / 100.0
    dt = (time_pairs[1] - time_pairs[0]) / 1000.0  # ms -> s

    # Convert pixels/frame -> m/s
    vR_full *= dR / dt
    vZ_full *= dZ / dt

    # Downsample to orig_res
    n_R, n_Z     = orig_res          # e.g. 8, 8
    res_x, res_y = vR_full.shape[2], vR_full.shape[1]   # W, H of interpolated images
    px = res_x // n_R
    py = res_y // n_Z

    R_down = np.array([R_interp[i * px:(i + 1) * px].mean() for i in range(n_R)])
    Z_down = np.array([Z_interp[j * py:(j + 1) * py].mean() for j in range(n_Z)])

    vR_down = np.zeros((vR_full.shape[0], n_Z, n_R), dtype=np.float32)
    vZ_down = np.zeros((vZ_full.shape[0], n_Z, n_R), dtype=np.float32)

    for j in range(n_Z):
        for i in range(n_R):
            vR_sub = vR_full[:, j * py:(j + 1) * py, i * px:(i + 1) * px]
            vZ_sub = vZ_full[:, j * py:(j + 1) * py, i * px:(i + 1) * px]
            vR_down[:, j, i] = vR_sub.mean(axis=(1, 2))
            vZ_down[:, j, i] = vZ_sub.mean(axis=(1, 2))

    R_profile  = R_interp.copy()
    vR_profile = vR_full.mean(axis=(0, 1))                # avg over time and Z
    vZ_profile = vZ_full.mean(axis=(0, 1))                # avg over time and Z
    # fluctuationg components of vR, vZ
    dvR = vR_full - vR_full.mean(axis=0)
    dvZ = vZ_full - vZ_full.mean(axis=0)
    # Reynolds stress <vR*vZ>
    RS_profile = (dvR * dvZ).mean(axis=(0, 1)) 
    # dn
    frames = frames[:-1]
    frames = frames - frames.mean(axis=0)
    # particle flux profile <vR*dn>
    flux_profile = (dvR * frames).mean(axis=(0, 1))   

    entry = {
        'label':                  label,
        'method_key':             method_key,
        'R_profile':              R_profile,
        'vR_profile':             vR_profile,
        'vZ_profile':             vZ_profile,
        'ReynoldsStress_profile': RS_profile,
        'flux_profile':           flux_profile,
    }

    # Keep only the frames requested for the quiver overlay
    if quiver_frames is not None and len(quiver_frames) > 0:
        idx = np.asarray(quiver_frames, dtype=int)
        entry['quiver_frames'] = idx
        entry['vR_frames']     = vR_full[idx].copy()
        entry['vZ_frames']     = vZ_full[idx].copy()

    # save to hdf5
    out_path = os.path.join(args.output_dir, f"{stem}_{method_key}.h5")
    save_result(out_path, vR_down, vZ_down, R_down, Z_down, time_pairs, entry)

    results_to_plot.append(entry)

    return results_to_plot


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Optical flow inference on experimental BES data'
    )
 
    # Input
    parser.add_argument('--input', required=True,
                        help='HDF5 file with BES images (keys: images, time, R, Z)')
    parser.add_argument('--output_dir', default=None,
                        help='Directory for output files '
                             '(default: same directory as input)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--load_saved', action='store_true',
                        help='Skip all inference and re-plot from the '
                             '<stem>_<method>.h5 files already in --output_dir')
 
    # Neural-net weights (optional — method is skipped when not provided and
    # not explicitly forced via the corresponding skip flag)
    parser.add_argument('--weights_pwc',      default=None,
                        help='Checkpoint for PWCNet')
    parser.add_argument('--weights_flownets', default=None,
                        help='Checkpoint for BESFlowNetS')
 
    # Skip flags
    parser.add_argument('--skip_pwc',       action='store_true')
    parser.add_argument('--skip_flownets',  action='store_true')
    parser.add_argument('--skip_raft',      action='store_true')
    parser.add_argument('--skip_farneback', action='store_true')
    parser.add_argument('--skip_odp',       action='store_true')
 
    # Output resolution (original BES grid)
    parser.add_argument('--orig_res_x', type=int, default=8,
                        help='Original BES resolution in R direction (default 8)')
    parser.add_argument('--orig_res_y', type=int, default=8,
                        help='Original BES resolution in Z direction (default 8)')
 
    # Plot flag
    parser.add_argument('--plot', action='store_true',
                        help='Plot velocity radial profiles after inference')
    parser.add_argument('--velocity_component', choices=['R', 'Z'], default='Z',
                        help="Velocity component to plot: 'R' for vR, 'Z' for vZ")

    # Quiver overlay plot
    parser.add_argument('--plot_quiver', action='store_true',
                        help='Plot selected frames with overlaid flow quiver '
                             '(one row per method, one column per frame)')
    parser.add_argument('--quiver_frames', type=int, nargs='+', default=None,
                        help='Pair indices to display (default: 4 evenly spaced)')
    parser.add_argument('--n_quiver_frames', type=int, default=4,
                        help='Number of evenly spaced frames when '
                             '--quiver_frames is not given (default 4)')
    parser.add_argument('--quiver_step', type=int, default=8,
                        help='Arrow sampling stride in pixels; arrows are taken '
                             'at arange(qs//2, H, qs) x arange(qs//2, W, qs). '
                             'Default 8 -> an 8x8 arrow grid on 64x64 images')
    parser.add_argument('--quiver_scale', type=float, default=None,
                        help='Quiver scale in (m/s) per cm of arrow length. '
                             'Default: auto, common to all panels')
    parser.add_argument('--quiver_flip_z', action='store_true',
                        help='Force Z to increase upward in the quiver plot. '
                             'By default the Z axis keeps the ordering of the '
                             'Z array in the input file, matching '
                             "imshow(origin='lower', extent=[R0,R1,Z0,Z1])")
 
    args   = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
 
    # ── Output directory ────────────────────────────────────────────────────
    if args.output_dir is None:
        # args.output_dir = os.path.dirname(os.path.abspath(args.input))
        args.output_dir = 'outputs/predicted_flows'
    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input))[0]
 
    # ── Load & preprocess ───────────────────────────────────────────────────
    print(f"\nLoading {args.input} ...")
    images, time_ax, R, Z = load_bes_h5(args.input)
    N = 3000
    images, time_ax = images[:N, :, :], time_ax[:N]
 
    if args.load_saved:
        # No inference: the frame pairs are never needed, only the raw images
        # (quiver background) and the time axis.
        n_pairs = len(images) - 1
        print(f"  {n_pairs} consecutive pairs (inference skipped)")
    else:
        # Neural nets: per-pair normalization
        print(f"Building frame pairs for neural nets ")
        framesA_norm, framesB_norm = make_pairs(images, per_pair_norm=True)

        # Classical methods: sequence-normalised pairs — joint [0,1] scale
        print("Normalizing sequence jointly (for classical methods)...")
        images_norm = normalize_sequence(images)
        framesA, framesB = make_pairs(images_norm)

        n_pairs = len(framesA)
        print(f"  {n_pairs} consecutive pairs")
 
    # Time axis for the velocity frames (time of frame A in each pair)
    time_pairs = time_ax[:n_pairs]
    orig_res   = (args.orig_res_x, args.orig_res_y)

    # ── Frames to keep for the quiver overlay ───────────────────────────────
    if args.plot_quiver and args.load_saved:
        # The full-resolution velocity fields only exist for whatever frames
        # were stored at inference time — the selection cannot be changed now.
        quiver_idx = None
    elif args.plot_quiver:
        if args.quiver_frames is not None:
            quiver_idx = np.array(sorted({int(i) for i in args.quiver_frames
                                          if 0 <= int(i) < n_pairs}), dtype=int)
            dropped = len(args.quiver_frames) - len(quiver_idx)
            if dropped:
                print(f"  [quiver] dropped {dropped} out-of-range frame index/indices")
        else:
            n_q = max(1, min(args.n_quiver_frames, n_pairs))
            quiver_idx = np.linspace(0, n_pairs - 1, n_q).astype(int)
        print(f"  Quiver frames: {quiver_idx.tolist()}")
    else:
        quiver_idx = None
 
    # ── Collect results for optional plotting ────────────────────────────────
    results_to_plot = []

    # ── 0. Load previously saved results instead of running anything ────────
    if args.load_saved:
        print("\n--- Loading saved results ---")
        for key in ('pwc', 'flownet', 'odp', 'raft', 'farneback'):
            in_path = os.path.join(args.output_dir, f"{stem}_{key}.h5")
            if not os.path.exists(in_path):
                continue
            entry = load_result(in_path, method_key=key)
            if entry is None:
                continue
            results_to_plot.append(entry)
            print(f"  Loaded {entry['label']:<12} <- {in_path}")

        if not results_to_plot:
            print(f"  No usable {stem}_<method>.h5 files found in {args.output_dir}")

        if args.plot_quiver:
            with_frames = [e for e in results_to_plot if 'vR_frames' in e]
            if not with_frames:
                print("  [quiver] saved files carry no stored frames — "
                      "re-run inference with --plot_quiver to write them")
                args.plot_quiver = False
            else:
                # Use the intersection so every row shows the same frames
                common = set(map(int, with_frames[0]['quiver_frames']))
                for e in with_frames[1:]:
                    common &= set(map(int, e['quiver_frames']))
                if not common:
                    print("  [quiver] saved files share no common frames — skipping")
                    args.plot_quiver = False
                else:
                    quiver_idx = np.array(sorted(common), dtype=int)
                    for e in with_frames:
                        sel = np.array([list(map(int, e['quiver_frames'])).index(i)
                                        for i in quiver_idx], dtype=int)
                        e['vR_frames'] = np.asarray(e['vR_frames'])[sel]
                        e['vZ_frames'] = np.asarray(e['vZ_frames'])[sel]
                    results_to_plot = with_frames
                    print(f"  Quiver frames: {quiver_idx.tolist()}")

    # ── 1. PWCNet ────────────────────────────────────────────────────────────
    if not args.load_saved and not args.skip_pwc:
        if args.weights_pwc is None:
            print("\n[PWC] --weights_pwc not provided — skipping")
        else:
            print("\n--- PWCNet ---")
            model = load_pwc(args.weights_pwc, device)
            t0 = time.perf_counter()
            flows = run_bes_model(model, framesA_norm, framesB_norm, device, args.batch_size)
            elapsed = time.perf_counter() - t0
            print(f"  Elapsed: {elapsed:.3f} s  ({elapsed * 1000 / n_pairs:.2f} ms/frame)")
            del model
            results_to_plot = postprocess_flows(flows, images, R, Z, time_pairs, orig_res, 
                                                results_to_plot, stem, 'pwc', 'PWC',
                                                quiver_frames=quiver_idx)
 
    # ── 2. BESFlowNetS ───────────────────────────────────────────────────────
    if not args.load_saved and not args.skip_flownets:
        if args.weights_flownets is None:
            print("\n[FlowNetS] --weights_flownets not provided — skipping")
        else:
            print("\n--- BESFlowNetS ---")
            model = load_flownets(args.weights_flownets, device)
            t0 = time.perf_counter()
            flows = run_bes_model(model, framesA_norm, framesB_norm, device, args.batch_size)
            elapsed = time.perf_counter() - t0
            print(f"  Elapsed: {elapsed:.3f} s  ({elapsed * 1000 / n_pairs:.2f} ms/frame)")
            del model
            results_to_plot = postprocess_flows(flows, images, R, Z, time_pairs, orig_res, 
                                                results_to_plot, stem, 'flownet', 'FlowNetS',
                                                quiver_frames=quiver_idx)
 
    # ── 3. ODP ───────────────────────────────────────────────────────────────
    if not args.load_saved and not args.skip_odp:
        print("\n--- ODP ---")
        flows, elapsed, ms_per_frame = run_odp(framesA_norm, framesB_norm)
        print(f"  Elapsed: {elapsed:.3f} s  ({ms_per_frame:.2f} ms/frame)")
        results_to_plot = postprocess_flows(flows, images, R, Z, time_pairs, orig_res, 
                                            results_to_plot, stem, 'odp', 'ODP',
                                            quiver_frames=quiver_idx)
    
    # ── 4. RAFT-small ────────────────────────────────────────────────────────
    if not args.load_saved and not args.skip_raft:
        print("\n--- RAFT-small ---")
        flows, elapsed, ms_per_frame = run_raft_small(framesA_norm, framesB_norm, device, args.batch_size)
        print(f"  Elapsed: {elapsed:.3f} s  ({ms_per_frame:.2f} ms/frame)")
        results_to_plot = postprocess_flows(flows, images, R, Z, time_pairs, orig_res, 
                                            results_to_plot, stem, 'raft', 'RAFT-small',
                                            quiver_frames=quiver_idx)
 
    # ── 5. Farneback ─────────────────────────────────────────────────────────
    if not args.load_saved and not args.skip_farneback:
        print("\n--- Farneback ---")
        flows, elapsed, ms_per_frame = run_farneback(framesA_norm, framesB_norm)
        print(f"  Elapsed: {elapsed:.3f} s  ({ms_per_frame:.2f} ms/frame)")
        results_to_plot = postprocess_flows(flows, images, R, Z, time_pairs, orig_res, 
                                            results_to_plot, stem, 'farneback', 'Farneback',
                                            quiver_frames=quiver_idx)
 
    # ── Plot ─────────────────────────────────────────────────────────────────
    if args.plot:
        if results_to_plot:
            plot_path = os.path.join(args.output_dir, f"{stem}_v{args.velocity_component.lower()}_profile.png")
            plot_v_profile(results_to_plot, velocity_component=args.velocity_component,
                           output_path=plot_path)
        else:
            print("\nNo results to plot — all methods were skipped.")

    if args.plot_quiver:
        if results_to_plot:
            quiver_path = os.path.join(args.output_dir, f"{stem}_flow_quiver.png")
            plot_flow_quivers(results_to_plot, images, R, Z, time_pairs, quiver_idx,
                              quiver_step=args.quiver_step, scale=args.quiver_scale,
                              flip_z=args.quiver_flip_z, cmap='RdBu_r',
                              output_path=quiver_path)
        else:
            print("\nNo results to plot — all methods were skipped.")
 
    print("\nDone.")
    