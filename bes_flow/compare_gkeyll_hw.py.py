# bes_flow/compare_gkeyll_hw.py
#
# Benchmark comparison of optical flow methods on Hasegawa-Wakatani
# turbulence simulated with Gkeyll.
#
# Motivation
# ──────────
# Consecutive Gkeyll frames are separated by a very small displacement
# (well below one pixel), so consecutive pairs carry almost no signal for
# an optical flow model.  Pairs are therefore built with a running window
# of `gap` frames:
#
#     pair i  =  (frame i, frame i + gap)      i = 0 .. Nframes-gap-1
#
# With Nframes = 250 and gap = 90 this yields 160 pairs.
#
# Ground truth
# ────────────
# Over a gap of ~85 frames the velocity field is NOT steady, so the
# single-step estimate  D = v(t_A) * dT  used for the JHTDB mixing layer
# is no longer valid.  The default here ('lagrangian') integrates the
# trajectory of every grid point forward through the intermediate
# velocity fields:
#
#     x_{k+1} = x_k + dt_k * v(t_k + dt_k/2,  x_k + dt_k/2 * v(t_k, x_k))
#     D(x_0)  = x_end - x_0
#
# with v interpolated linearly in time between stored frames and
# bilinearly in space.  This matches the convention of
# bes_flow.dataset.integrate_forward_displacement:
#
#     channel 0 = dx, channel 1 = dy,  and  frameA(x) ≈ frameB(x + D(x))
#
# '--gt_mode frozen' reproduces the old single-step behaviour
# (D = v(t_A) * dT) and is useful only as a sanity baseline.
#
# Gkeyll HDF5 file layout (as produced by the HW runs)
# ────────────────────────────────────────────────────
#   fields/density_fluctuation  (Nrows, W, H)     normalised density
#   fields/velocity             (Nrows, 2, W, H)  (vx, vy)
#   kappa_hw                    (Nrows,)          kappa of each row
#   kappa_hw_step_values        (Nkappa,)         unique kappa, ascending
#   x_cm                        (W,)              x grid [cm]
#   y_cm                        (H,)              y grid [cm]
#   time_s                      (Nrows,)          time [s]
#
# Rows belonging to one kappa form one contiguous time sequence; a single
# file holds many such sequences (one per kappa).
#
# Usage
# ─────
#   # Pick a gap: scan candidate gaps and print displacement statistics
#   python -m bes_flow.compare_gkeyll_hw \
#       --h5 synthetic_data/gkeyll_hw/near_zonal/moderate_power_nt_shot193813_t3875ms_Ln_0p03_to_0p8_m.h5 \
#       --kappa_index 20 --scan_gaps 20 40 60 80 90 100 120
#
#   # Evaluate models on one raw Gkeyll sequence
#   python -m bes_flow.compare_gkeyll_hw \
#       --h5 synthetic_data/gkeyll_hw/near_zonal/moderate_power_nt_shot193813_t3875ms_Ln_0p03_to_0p8_m.h5 \
#       --kappa_index 20 --gap 90 \
#       --weights_pwc      checkpoints/pwc_best.pt \
#       --weights_flownets checkpoints/flownets_best.pt \
#       --output outputs/gkeyll_hw/ --skip_odp
#
#   # Evaluate on the held-out test split of a prepared cache
#   python -m bes_flow.compare_gkeyll_hw \
#       --cache synthetic_data/gkeyll_hw/gkeyll_hw_train.h5 \
#       --weights_pwc checkpoints/pwc_best.pt --skip_odp

import os
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, map_coordinates

import torch

# ── Reuse all runner functions and reporting from compare_methods ──────────
from bes_flow.compare_methods import (
    load_pwc, load_flownets,
    run_bes_model, run_farneback, run_raft_small, run_odp,
    print_comparison_table, plot_metric_bars, plot_comparison_examples,
)
from bes_flow.compare_jhopkins import plot_animation
from bes_flow.dataset import BESDataset, load_dataset_cache, warp_image
from bes_flow.metrics import compute_all_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def list_kappas(h5_path):
    """
    Return (kappa_values, n_frames_per_kappa) for a Gkeyll HW file.

    kappa_values        : (Nkappa,) float — ascending turbulence drive
    n_frames_per_kappa  : (Nkappa,) int   — frames available for each kappa
    """
    with h5py.File(h5_path, 'r') as f:
        kappa_values = f['kappa_hw_step_values'][()]
        kappa_all    = f['kappa_hw'][()]
    counts = np.array([int(np.sum(kappa_all == k)) for k in kappa_values])
    return kappa_values, counts


def _to_yx(arr, nx, ny, frame_axes):
    """
    Bring the trailing two axes of `arr` into (..., ny, nx) order.

    Gkeyll stores fields as (..., W, H) = (..., x, y); the BES pipeline
    indexes [row, col] = [y, x].  When nx != ny the layout is detected
    automatically and `frame_axes` is only used to break the square tie.
    """
    if arr.shape[-2] == ny and arr.shape[-1] == nx and nx != ny:
        return arr                              # already (..., y, x)
    if arr.shape[-2] == nx and arr.shape[-1] == ny and nx != ny:
        return np.swapaxes(arr, -1, -2)         # (..., x, y) -> (..., y, x)
    if arr.shape[-2] != arr.shape[-1]:
        raise ValueError(
            f"Field shape {arr.shape} matches neither (ny={ny}, nx={nx}) "
            f"nor (nx={nx}, ny={ny})."
        )
    # Square grid — trust the flag
    return arr if frame_axes == 'yx' else np.swapaxes(arr, -1, -2)


def _resample_stack(stack, ny_new, nx_new):
    """
    Bilinearly resample a (N, ny, nx) stack onto an (ny_new, nx_new) grid,
    preserving the physical extent (corner pixels map to corner pixels).
    """
    N, ny, nx = stack.shape
    yy = np.linspace(0, ny - 1, ny_new, dtype=np.float32)
    xx = np.linspace(0, nx - 1, nx_new, dtype=np.float32)
    y2d, x2d = np.meshgrid(yy, xx, indexing='ij')
    coords = [y2d.ravel(), x2d.ravel()]
    out = np.empty((N, ny_new, nx_new), dtype=np.float32)
    for i in range(N):
        out[i] = map_coordinates(stack[i], coords, order=1,
                                 mode='nearest').reshape(ny_new, nx_new)
    return out


def load_gkeyll_h5(h5_path, kappa_index=None, kappa_value=None,
                   frame_axes='xy', crop=None, resize=None,
                   verbose=True):
    """
    Load one kappa sequence from a Gkeyll HW HDF5 file.

    Parameters
    ----------
    h5_path     : str
    kappa_index : int or None — index into kappa_hw_step_values (ascending)
    kappa_value : float or None — exact kappa value (overrides kappa_index)
    frame_axes  : 'xy' or 'yx' — spatial axis order of the stored fields.
                  Only consulted for square grids; otherwise auto-detected.
    crop        : (y0, y1, x0, x1) or None — pixel window applied before resize
    resize      : (ny, nx) or None — target grid, e.g. (64, 64)

    Returns
    -------
    images   : (Nframes, ny, nx) float32 — density fluctuation
    vx, vy   : (Nframes, ny, nx) float32 — velocity [cm/s]
    times    : (Nframes,)        float64 — time [s]
    x_points : (nx,)             float64 — x grid [cm]
    y_points : (ny,)             float64 — y grid [cm]
    """

    if verbose:
        print(f"Loading Gkeyll HW data from: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        kappa_values = f['kappa_hw_step_values'][()]
        kappa_all    = f['kappa_hw'][()]

        if kappa_value is None:
            if kappa_index is None:
                raise ValueError("Provide either kappa_index or kappa_value.")
            kappa_value = kappa_values[kappa_index]

        rows = np.flatnonzero(kappa_all == kappa_value)
        if rows.size == 0:
            raise ValueError(
                f"No rows with kappa == {kappa_value}. "
                f"Available: {np.asarray(kappa_values)}"
            )

        images = f['fields']['density_fluctuation'][rows].astype(np.float32)
        vel    = f['fields']['velocity'][rows].astype(np.float32)
        times  = f['time_s'][rows].astype(np.float64)
        x_points = f['x_cm'][()].astype(np.float64)
        y_points = f['y_cm'][()].astype(np.float64)

    nx, ny = len(x_points), len(y_points)

    images = _to_yx(images, nx, ny, frame_axes)
    vel    = _to_yx(vel,    nx, ny, frame_axes)
    vx, vy = vel[:, 0].copy(), vel[:, 1].copy()

    # Velocity units -> cm/s
    scale = 1.
    vx *= scale
    vy *= scale

    # Frames must be time-ordered for the trajectory integration
    order = np.argsort(times)
    if not np.all(order == np.arange(len(times))):
        print("  [warn] rows were not time-ordered — sorting by time_s")
        images, vx, vy, times = images[order], vx[order], vy[order], times[order]

    if crop is not None:
        y0, y1, x0, x1 = crop
        images = images[:, y0:y1, x0:x1]
        vx, vy = vx[:, y0:y1, x0:x1], vy[:, y0:y1, x0:x1]
        x_points, y_points = x_points[x0:x1], y_points[y0:y1]

    if resize is not None:
        ny_new, nx_new = resize
        if (images.shape[1], images.shape[2]) != (ny_new, nx_new):
            images = _resample_stack(images, ny_new, nx_new)
            vx     = _resample_stack(vx,     ny_new, nx_new)
            vy     = _resample_stack(vy,     ny_new, nx_new)
            x_points = np.linspace(x_points[0], x_points[-1], nx_new)
            y_points = np.linspace(y_points[0], y_points[-1], ny_new)

    dts = np.diff(times)
    if verbose:
        print(f"  kappa     : {float(kappa_value):.5g} "
              f"(index {kappa_index if kappa_index is not None else '-'} "
              f"of {len(kappa_values)})")
        print(f"  Frames    : {images.shape[0]}")
        print(f"  Grid      : {images.shape[2]} x {images.shape[1]}  (nx x ny)")
        print(f"  X-range   : {x_points[0]:.2f} - {x_points[-1]:.2f} cm")
        print(f"  Y-range   : {y_points[0]:.2f} - {y_points[-1]:.2f} cm")
        print(f"  Time span : {times[0]:.6g} - {times[-1]:.6g} s  "
              f"(dt = {dts.mean():.3g} s)")
        if dts.size and dts.std() > 1e-3 * abs(dts.mean()):
            print(f"  [warn] non-uniform frame spacing "
                  f"(dt std/mean = {dts.std()/abs(dts.mean()):.2%})")
        vmag = np.sqrt(vx ** 2 + vy ** 2)
        print(f"  |v|       : mean {vmag.mean():.3g}  max {vmag.max():.3g} cm/s")

    return images, vx, vy, times, x_points, y_points


# ─────────────────────────────────────────────────────────────────────────────
# Frame-pair and GT-flow construction
# ─────────────────────────────────────────────────────────────────────────────

def _sample_vel_px(vel_px, k_lo, w_lo, y, x):
    """
    Sample the velocity field (in px/s) at fractional time and position.

    vel_px : (Nframes, 2, ny, nx) — velocity in pixels per second
    k_lo   : int   — index of the earlier bracketing frame
    w_lo   : float — weight of that frame (1 -> exactly frame k_lo)
    y, x   : (ny, nx) float arrays of pixel coordinates
    """
    k_hi = min(k_lo + 1, vel_px.shape[0] - 1)
    coords = [y.ravel(), x.ravel()]

    def _interp(c):
        lo = map_coordinates(vel_px[k_lo, c], coords, order=1, mode='nearest')
        if k_hi == k_lo or w_lo >= 1.0:
            return lo.reshape(y.shape)
        hi = map_coordinates(vel_px[k_hi, c], coords, order=1, mode='nearest')
        return (w_lo * lo + (1.0 - w_lo) * hi).reshape(y.shape)

    return _interp(0), _interp(1)


def _lagrangian_displacement(vel_px, times, i_start, gap, n_substeps=1):
    """
    Integrate grid points forward from frame i_start to frame i_start+gap
    through the time-varying velocity field (RK2 midpoint per sub-step).

    Returns
    -------
    flow : (2, ny, nx) float32 — total pixel displacement (dx, dy)
    """
    _, _, ny, nx = vel_px.shape
    y0, x0 = np.meshgrid(np.arange(ny, dtype=np.float32),
                         np.arange(nx, dtype=np.float32), indexing='ij')
    x, y = x0.copy(), y0.copy()

    for k in range(i_start, i_start + gap):
        dt_frame = float(times[k + 1] - times[k])
        h = dt_frame / n_substeps
        for s in range(n_substeps):
            # Fractional position within frame interval k
            f0 = s / n_substeps
            fm = (s + 0.5) / n_substeps
            vx1, vy1 = _sample_vel_px(vel_px, k, 1.0 - f0, y, x)
            x_mid = x + 0.5 * h * vx1
            y_mid = y + 0.5 * h * vy1
            vxm, vym = _sample_vel_px(vel_px, k, 1.0 - fm, y_mid, x_mid)
            x = x + h * vxm
            y = y + h * vym

    return np.stack([x - x0, y - y0], axis=0).astype(np.float32)


def build_pairs_gap(images, vx, vy, times, x_points, y_points,
                    gap=90, psf_fwhm=None, gt_mode='frozen', #'lagrangian',
                    n_substeps=1, stride=1, normalize='minmax',
                    clip_pct=(0.5, 99.5), verbose=True):
    """
    Build running-window frame pairs (i, i+gap) and the matching GT flow.

    Convention (matching the BES pipeline):
      channel 0 = dx (x-direction), channel 1 = dy (y-direction),
      D lives at frame-A grid points:  frameA(x) ≈ frameB(x + D(x))

    Parameters
    ----------
    images     : (Nframes, ny, nx) float32
    vx, vy     : (Nframes, ny, nx) float32 — velocity [cm/s]
    times      : (Nframes,) float64 — [s]
    x_points   : (nx,) — x grid [cm]
    y_points   : (ny,) — y grid [cm]
    gap        : int — frame separation of each pair
    psf_fwhm   : float or None — FWHM (px) of an isotropic Gaussian applied
                 to images and velocities to mimic the BES point-spread
                 function.  sigma = fwhm / (2*sqrt(2*ln2)) ≈ fwhm / 2.355
    gt_mode    : 'lagrangian' — integrate trajectories through the whole gap
                 'frozen'     — single Euler step, D = v(t_A) * dT
    n_substeps : RK2 sub-steps per frame interval (1 is usually plenty for
                 a large gap; raise it if the per-frame displacement
                 approaches a pixel)
    stride     : take every `stride`-th pair (1 = fully overlapping window)
    normalize  : 'minmax' (global min/max, as in compare_jhopkins) or
                 'percentile' (robust clip to clip_pct then rescale)

    Returns
    -------
    framesA  : (N_pairs, 1, ny, nx) float32 in [0, 1]
    framesB  : (N_pairs, 1, ny, nx) float32 in [0, 1]
    flows_gt : (N_pairs, 2, ny, nx) float32 — pixel displacement
    """
    Nframes, ny, nx = images.shape
    if gap >= Nframes:
        raise ValueError(f"gap={gap} but only {Nframes} frames available.")

    starts  = np.arange(0, Nframes - gap, stride)
    N_pairs = len(starts)

    dx_phys = (x_points[-1] - x_points[0]) / (nx - 1)   # cm per pixel
    dy_phys = (y_points[-1] - y_points[0]) / (ny - 1)

    if psf_fwhm is not None:
        psf_sigma = psf_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        if verbose:
            print(f"  Applying BES PSF (FWHM={psf_fwhm:.1f} px, "
                  f"sigma={psf_sigma:.2f} px) to images and velocities...")
        images = np.stack([gaussian_filter(im, sigma=psf_sigma) for im in images])
        vx     = np.stack([gaussian_filter(v,  sigma=psf_sigma) for v in vx])
        vy     = np.stack([gaussian_filter(v,  sigma=psf_sigma) for v in vy])

    # Normalise the whole stack to [0, 1]
    if normalize == 'percentile':
        lo, hi = np.percentile(images, clip_pct)
    else:
        lo, hi = float(images.min()), float(images.max())
    images_norm = ((images - lo) / (hi - lo)).astype(np.float32) if hi > lo \
        else images.astype(np.float32)
    images_norm = np.clip(images_norm, 0.0, 1.0)

    # Velocity in pixels per second
    vel_px = np.stack([vx / dx_phys, vy / dy_phys], axis=1).astype(np.float32)

    framesA  = np.zeros((N_pairs, 1, ny, nx), dtype=np.float32)
    framesB  = np.zeros((N_pairs, 1, ny, nx), dtype=np.float32)
    flows_gt = np.zeros((N_pairs, 2, ny, nx), dtype=np.float32)

    if verbose:
        print(f"  Building {N_pairs} pairs (gap={gap}, stride={stride}, "
              f"gt_mode='{gt_mode}')...")

    for j, i in enumerate(starts):
        framesA[j, 0] = images_norm[i]
        framesB[j, 0] = images_norm[i + gap]

        if gt_mode == 'frozen':
            dT = float(times[i + gap] - times[i])
            flows_gt[j, 0] = vel_px[i, 0] * dT
            flows_gt[j, 1] = vel_px[i, 1] * dT
        elif gt_mode == 'lagrangian':
            flows_gt[j] = _lagrangian_displacement(vel_px, times, i, gap,
                                                   n_substeps=n_substeps)
        else:
            raise ValueError("gt_mode must be 'lagrangian' or 'frozen'")

        if verbose and (j + 1) % max(1, N_pairs // 5) == 0:
            print(f"    {j+1}/{N_pairs} pairs")

    if verbose:
        dT = float(times[gap] - times[0])
        mag = np.sqrt(flows_gt[:, 0] ** 2 + flows_gt[:, 1] ** 2)
        print(f"\nBuilt {N_pairs} frame pairs, dT = {dT:.4g} s per pair")
        print(f"  GT displacement — mean: {mag.mean():.3f} px  "
              f"max: {mag.max():.3f} px  std: {mag.std():.3f} px")
        print(f"  Pixel size      — dx: {dx_phys:.4g} cm  dy: {dy_phys:.4g} cm")

    return framesA, framesB, flows_gt


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics — do the pairs actually contain the motion the GT claims?
# ─────────────────────────────────────────────────────────────────────────────

def estimate_bulk_shift(fA, fB):
    """
    Integer-pixel bulk shift (dx, dy) from FFT cross-correlation of two
    mean-subtracted frames.  A cheap, model-free cross-check on the GT:
    a wrong velocity unit or a transposed axis shows up immediately.
    """
    a = fA - fA.mean()
    b = fB - fB.mean()
    corr = np.fft.ifft2(np.fft.fft2(b) * np.conj(np.fft.fft2(a))).real
    ny, nx = corr.shape
    iy, ix = np.unravel_index(np.argmax(corr), corr.shape)
    dy = iy - ny if iy > ny // 2 else iy
    dx = ix - nx if ix > nx // 2 else ix
    return float(dx), float(dy)


def check_pairs(framesA, framesB, flows_gt, n_check=20, verbose=True):
    """
    Report how well the ground truth explains each pair.

    - warp consistency : correlation of warp(A, flow) with B, against the
      raw correlation of A with B.  Higher after warping means the GT
      displacement really is the motion in the images.
    - bulk shift       : cross-correlation peak vs. the GT mean vector.

    Returns a dict of summary statistics.
    """
    N = len(framesA)
    idx = np.linspace(0, N - 1, min(n_check, N)).astype(int)

    corr_raw, corr_warp, shift_err = [], [], []
    for i in idx:
        A, B, D = framesA[i, 0], framesB[i, 0], flows_gt[i]
        warped = warp_image(A, D)
        corr_raw.append(np.corrcoef(A.ravel(), B.ravel())[0, 1])
        corr_warp.append(np.corrcoef(warped.ravel(), B.ravel())[0, 1])
        dx_xc, dy_xc = estimate_bulk_shift(A, B)
        shift_err.append((dx_xc - D[0].mean(), dy_xc - D[1].mean()))

    corr_raw  = np.array(corr_raw)
    corr_warp = np.array(corr_warp)
    shift_err = np.array(shift_err)
    mag = np.sqrt(flows_gt[:, 0] ** 2 + flows_gt[:, 1] ** 2)

    stats = {
        'corr_raw':      float(corr_raw.mean()),
        'corr_warped':   float(corr_warp.mean()),
        'gain':          float(corr_warp.mean() - corr_raw.mean()),
        'shift_err_px':  float(np.abs(shift_err).mean()),
        'mean_disp_px':  float(mag.mean()),
        'max_disp_px':   float(mag.max()),
    }

    if verbose:
        print("\nPair quality check "
              f"({len(idx)} of {N} pairs)")
        print(f"  corr(A, B)                 : {stats['corr_raw']:+.3f}")
        print(f"  corr(warp(A, D_gt), B)     : {stats['corr_warped']:+.3f}   "
              f"(gain {stats['gain']:+.3f})")
        print(f"  |bulk XC shift - mean D|   : {stats['shift_err_px']:.2f} px")
        print(f"  GT displacement            : mean {stats['mean_disp_px']:.2f} px, "
              f"max {stats['max_disp_px']:.2f} px")
        if stats['gain'] < 0.02:
            print("  [warn] warping by the GT barely improves the match — the "
                  "gap may be long enough that structures decorrelate, or the "
                  "velocity units / axis order may be wrong.")
        if stats['shift_err_px'] > 3.0 and stats['mean_disp_px'] > 1.0:
            print("  [warn] cross-correlation shift disagrees with the GT — "
                  "check --frame_axes.")
    return stats


def scan_gaps(images, vx, vy, times, x_points, y_points, gaps,
              psf_fwhm=None, gt_mode='frozen', n_check=8):
    """
    Print displacement and correlation statistics for several candidate
    gaps so a sensible window can be chosen before building a dataset.

    'frozen' GT is used by default here because it is ~100x faster and
    good enough for ranking gaps; re-check the chosen gap with
    --gt_mode lagrangian.
    """
    print("\nGap scan")
    print(f"{'gap':>5} {'pairs':>6} {'mean|D|':>9} {'max|D|':>8} "
          f"{'corr(A,B)':>10} {'corr warp':>10}")
    print("-" * 54)
    rows = []
    for gap in gaps:
        if gap >= len(images):
            continue
        fA, fB, fl = build_pairs_gap(
            images, vx, vy, times, x_points, y_points, gap=gap,
            psf_fwhm=psf_fwhm, gt_mode=gt_mode, verbose=False,
        )
        st = check_pairs(fA, fB, fl, n_check=n_check, verbose=False)
        rows.append((gap, len(fA), st))
        print(f"{gap:>5} {len(fA):>6} {st['mean_disp_px']:>9.2f} "
              f"{st['max_disp_px']:>8.2f} {st['corr_raw']:>10.3f} "
              f"{st['corr_warped']:>10.3f}")
    print("\nPick the largest gap that still has a clear positive "
          "corr(warp) - corr(A,B) gain and a mean |D| inside the "
          "displacement range the models were trained on.")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Compare optical flow methods on Gkeyll Hasegawa-Wakatani data'
    )

    # ── Data source (mutually exclusive) ──────────────────────────────────
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--h5', metavar='FILE',
                        help='Raw Gkeyll HW HDF5 file')
    source.add_argument('--cache', metavar='FILE',
                        help='Prepared dataset cache; its test split is used')

    # ── Sequence selection (raw mode) ─────────────────────────────────────
    parser.add_argument('--kappa_index', type=int, default=20,
                        help='Index into kappa_hw_step_values (default: 20)')
    parser.add_argument('--list_kappas', action='store_true',
                        help='Print available kappa values and exit')
    parser.add_argument('--frame_axes', choices=['xy', 'yx'], default='xy',
                        help='Spatial axis order of stored fields; only used '
                             'to break the tie on a square grid (default: xy)')
    parser.add_argument('--crop', type=int, nargs=4, default=None,
                        metavar=('Y0', 'Y1', 'X0', 'X1'),
                        help='Pixel window applied before resizing')
    parser.add_argument('--resize', type=int, nargs=2, default=None,
                        metavar=('NY', 'NX'),
                        help='Resample to this grid, e.g. --resize 64 64')

    # ── Pair construction ─────────────────────────────────────────────────
    parser.add_argument('--gap', type=int, default=90,
                        help='Frame separation within a pair (default: 90)')
    parser.add_argument('--stride', type=int, default=1,
                        help='Step between pair start frames (default: 1)')
    parser.add_argument('--gt_mode', choices=['lagrangian', 'frozen'],
                        default='lagrangian')
    parser.add_argument('--n_substeps', type=int, default=1,
                        help='RK2 sub-steps per frame interval (default: 1)')
    parser.add_argument('--psf_fwhm', type=float, default=None,
                        help='BES PSF FWHM in pixels (default: none)')
    parser.add_argument('--normalize', choices=['minmax', 'percentile'],
                        default='minmax')
    parser.add_argument('--scan_gaps', type=int, nargs='+', default=None,
                        metavar='GAP',
                        help='Report statistics for these gaps and exit')

    # ── Output ────────────────────────────────────────────────────────────
    parser.add_argument('--output', default=None,
                        help='Directory for figures and results')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_examples', type=int, default=5)

    # ── Neural net weights ────────────────────────────────────────────────
    parser.add_argument('--weights_pwc',      default=None)
    parser.add_argument('--weights_flownets', default=None)

    # ── Animation ─────────────────────────────────────────────────────────
    parser.add_argument('--plot_ani', action='store_true')

    # ── Skip flags ────────────────────────────────────────────────────────
    parser.add_argument('--skip_pwc',       action='store_true')
    parser.add_argument('--skip_flownets',  action='store_true')
    parser.add_argument('--skip_odp',       action='store_true')
    parser.add_argument('--skip_farneback', action='store_true')
    parser.add_argument('--skip_raft',      action='store_true')

    args   = parser.parse_args()

    if args.list_kappas:
        kvals, counts = list_kappas(args.h5)
        print(f"{'idx':>4} {'kappa':>12} {'frames':>8}")
        for i, (k, c) in enumerate(zip(kvals, counts)):
            print(f"{i:>4} {float(k):>12.5g} {c:>8}")
        raise SystemExit(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    images = times = vx = vy = x_points = y_points = None

    # ── Load data and build pairs ─────────────────────────────────────────
    if args.cache:
        print(f"Loading prepared cache: {args.cache}")
        (_, _, _, _, _, _,
         framesA, framesB, flows_gt, metadata) = load_dataset_cache(args.cache)
        print(f"  Test split: {len(framesA)} pairs, shape {framesA.shape[1:]}")
        for k, v in metadata.items():
            print(f"    {k}: {v}")
    else:
        images, vx, vy, times, x_points, y_points = load_gkeyll_h5(
            args.h5,
            kappa_index = args.kappa_index,
            frame_axes  = args.frame_axes,
            crop        = tuple(args.crop) if args.crop else None,
            resize      = tuple(args.resize) if args.resize else None,
        )

        if args.scan_gaps:
            scan_gaps(images, vx, vy, times, x_points, y_points,
                      args.scan_gaps, psf_fwhm=args.psf_fwhm)
            raise SystemExit(0)

        framesA, framesB, flows_gt = build_pairs_gap(
            images, vx, vy, times, x_points, y_points,
            gap        = args.gap,
            psf_fwhm   = args.psf_fwhm,
            gt_mode    = args.gt_mode,
            n_substeps = args.n_substeps,
            stride     = args.stride,
            normalize  = args.normalize,
        )

    check_pairs(framesA, framesB, flows_gt)

    test_dataset = BESDataset(framesA, framesB, flows_gt, augment=False)

    # ── Run methods ───────────────────────────────────────────────────────
    all_flows = {}
    all_times = {}   # {method: algorithm-only wall-clock seconds}

    # 1. PWCNet
    if not args.skip_pwc:
        if args.weights_pwc is None:
            print("\n  [PWC] --weights_pwc not provided — skipping")
        else:
            print("\nPWCNet:")
            model_pwc = load_pwc(args.weights_pwc, device)
            all_flows['PWC'], elapsed, ms_pf = run_bes_model(
                model_pwc, test_dataset, device, args.batch_size
            )
            all_times['PWC'] = (elapsed, ms_pf)
            del model_pwc
            print(f'  Elapsed time {elapsed:.3f} s')

    # 2. BESFlowNetS
    if not args.skip_flownets:
        if args.weights_flownets is None:
            print("\n  [FlowNetS] --weights_flownets not provided — skipping")
        else:
            print("\nBESFlowNetS:")
            model_f = load_flownets(args.weights_flownets, device)
            all_flows['FlowNetS'], elapsed, ms_pf = run_bes_model(
                model_f, test_dataset, device, args.batch_size
            )
            all_times['FlowNetS'] = (elapsed, ms_pf)
            del model_f
            print(f'  Elapsed time {elapsed:.3f} s')

    # 3. ODP
    if not args.skip_odp:
        all_flows['ODP'], elapsed, ms_pf = run_odp(framesA, framesB)
        all_times['ODP'] = (elapsed, ms_pf)
        print(f'  Elapsed time {elapsed:.3f} s')

    # 4. Farneback
    if not args.skip_farneback:
        all_flows['Farneback'], elapsed, ms_pf = run_farneback(framesA, framesB)
        all_times['Farneback'] = (elapsed, ms_pf)
        print(f'  Elapsed time {elapsed:.3f} s')

    # 5. RAFT-small
    if not args.skip_raft:
        all_flows['RAFT-small'], elapsed, ms_pf = run_raft_small(
            framesA, framesB, device, args.batch_size
        )
        all_times['RAFT-small'] = (elapsed, ms_pf)
        print(f'  Elapsed time {elapsed:.3f} s')

    if args.plot_ani and images is not None:
        plot_animation(images, times, vx, vy, x_points, y_points,
                       save_ani=True)

    if not all_flows:
        print("No methods were run — nothing to compare.")
        raise SystemExit(0)

    # ── Metrics ───────────────────────────────────────────────────────────
    print("\nComputing metrics...")
    all_results = {}
    for method, flows_pred in all_flows.items():
        print(f"  {method}")
        all_results[method] = compute_all_metrics(flows_pred, flows_gt)

    # ── Summary table ─────────────────────────────────────────────────────
    print_comparison_table(all_results, all_times)

    # ── Figures ───────────────────────────────────────────────────────────
    print("Plotting figures...")
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    plot_metric_bars(all_results, all_times, output_dir=args.output)
    plot_comparison_examples(framesA, framesB, flows_gt, all_flows,
                             n_examples=args.n_examples,
                             output_dir=args.output,
                             quiver_scale=30)

    print("\nDone.")
