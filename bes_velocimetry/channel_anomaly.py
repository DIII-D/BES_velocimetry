from bes_velocimetry import create_structure
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, jit, cuda
import scipy.signal as signal
import scipy.stats as stats
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

@jit
def _estimate_snr_core(x, smoothed):
    noise = x - smoothed
    signal_power = np.mean(smoothed ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return np.inf if signal_power > 0 else 0
    return 10 * np.log10(signal_power / noise_power)

def _estimate_snr(x, window_length=51):
    # Smooth signal (Savitzky-Golay filter for noise suppression)
    smoothed = signal.savgol_filter(x, window_length, polyorder=3)
    return _estimate_snr_core(x, smoothed)

@jit
def get_neighbors_grid(i, j, H=8, W=8):
    # Define neighbor offsets for 8-connected neighbors
    neighbor_offsets = [(-1,-1), (-1,0), (-1,1),
                        (0,-1),         (0,1),
                        (1,-1),  (1,0), (1,1)]
    indices = []
    for di, dj in neighbor_offsets:
        ni, nj = i + di, j + dj
        if (ni < 0) or (ni >= H) or (nj < 0) or (nj >= W):
            continue
        indices.append((ni * W) + nj)
    return np.array(indices)

@jit
def get_neighbors_raw(index,H=8, W=8):
    x, y = (index % W), (index // W)
    return get_neighbors_grid(y, x, H, W)

@jit
def get_neighbors_value(data, index):
    neighbor_channels = get_neighbors_raw(index)
    sampled_data = data[neighbor_channels,:]
    return sampled_data.transpose()

def generate_t_dist(n_total, alpha):
    #alpha = 0.05  # Grubbs significance level
    result = []
    for n in range(n_total):
        #result.append(stats.t.ppf(1 - alpha / (2 * (n + 0)), df=n - 2))
        try:
            result.append(stats.t.ppf(1 - alpha / (2 * n), df=n - 2))
        except:
            result.append(float('nan'))
    return np.array(result)

def generate_g_crit(n_total, alpha):
    t_dist = generate_t_dist(n_total, alpha)
    result = []
    for n in range(n_total+1):
        try:
            t_dist_entry = t_dist[n]
            result.append(((n - 1) / np.sqrt(n)) * np.sqrt(t_dist_entry**2 / (n - 2 + t_dist_entry**2)))
        except:
            result.append(float('nan'))
    return result

@jit(nopython=False)
def spatial_grubbs_pixel_core(data, i, j, G_crit_array, t_window=10, threshold=0.25):
    T = data.shape[1]
    H, W = 8, 8
    # For each pixel, track how often it's a spatial outlier over time
    outlier_counts = 0
    print(i, j)
    index = (i * W) + j
    neighbors = get_neighbors_value(data, index)
    for t in range(T):
        center_val = data[index][t]
        # Grubbs-style test: is center_val an outlier among neighbors?
        neighbors_frame = neighbors[t:(t + t_window),:].ravel()
        if (len(neighbors_frame) < (3*t_window)):
            continue
        mean = np.mean(neighbors_frame)
        std = np.std(neighbors_frame)
        # print(std)
        if std == 0:
            continue
        G = abs(center_val - mean) / std
        n = len(neighbors_frame)
        #t_dist_entry = t_dist[n - 1]
        #alpha = 0.01
        #t_dist_entry = stats.t.ppf(1 - alpha / (2 * n), df=n - 2)
        #G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_dist_entry**2 / (n - 2 + t_dist_entry**2))
        G_crit = G_crit_array[n]
        if G > G_crit:
            outlier_counts += 1    
    return index, outlier_counts >= (threshold * T), i, j, outlier_counts

class ChannelAnomaly:
    def __init__(self):
        pass

    # Gives a test case and the structure of the frames expected.
    # We expect (channel, T) as the test case data structures and for production
    # Return a time series that is T/10000 seconds long - assuming we are on the slow data which is 10khz
    def createSimulatedTestCase(self):
        T, H, W = 100000, 8, 8
        np.random.seed(34821384)
        frames = np.random.normal(loc=0, scale=1, size=(H*W, T))

        # Inject bad channels (anomalous behavior)
        frames[0,:] = -4      # hot
        frames[1,:] += -4      # stuck low
        frames[2,:] += np.sin(np.linspace(0, T, T)) * 50  # oscillatory artifact
        frames[55,:] = 4      # hot
        frames[63,:] -= np.sin(np.linspace(0, T, T)) * 50  # oscillatory artifact

        tframes_base = np.linspace(0, T/10000, T)
        tframes = np.zeros(((H*W), T))
        for n in range(H*W):
            tframes[n] = tframes_base

        return frames, tframes
    
    # Frames and tframes are of shape (64, T)
    def filterTimeSeries(self, frames, tframes, xr):
        mask = np.where((tframes[0,:]<=xr[1])&(tframes[0,:]>=xr[0]))[0]
        frames_filtered = frames[:,mask]
        tframes_filtered = tframes[:, mask]
        return frames_filtered, tframes
    
    # Anomaly detection mechanisms return a numpy list of indices of anomalies of (H,W) shape.
    # We also return the relevant scores for the method of the same shape.

    def isolationForest(self, frames, H=8, W=8, contamination=0.05, estimators=10000, verbose=False):
        T = frames.shape[-1]
        # --- Reshape: each channel = time series feature vector ---
        X = frames.reshape(H * W,T)  # shape: (64, T) → each row is a channel's time series

        # --- Train Isolation Forest ---
        clf = IsolationForest(
            contamination=contamination,
            n_estimators=estimators
        )
        pred = clf.fit_predict(X)  # -1 = outlier, +1 = normal

        # --- Post-process ---
        bad_mask = (pred == -1).reshape(H, W)

        # Optional: outlier scores (lower = more anomalous)
        scores = clf.decision_function(X)  # shape: (64,)
        scores_grid = scores.reshape(H, W)
        if verbose:

            # --- Plot Results ---
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(np.mean(frames, axis=-1).reshape(H,W), cmap='viridis')
            plt.title("Mean Frame")
            plt.colorbar()

            plt.subplot(1, 3, 2)
            plt.imshow(scores_grid, cmap='plasma')
            plt.title("Isolation Forest Scores")
            plt.colorbar(label="Higher = more normal")

            plt.subplot(1, 3, 3)
            plt.imshow(bad_mask, cmap='gray')
            plt.title("Detected Bad Channels")
            plt.colorbar(label="Bad=True")

            plt.tight_layout()
            plt.show()

            print("Bad channel indices (row, col):", np.argwhere(bad_mask))
        return np.argwhere(bad_mask), scores_grid
    
    def pca(self, frames, H=8, W=8, components=5, top_error=5, verbose=False):
        T = frames.shape[-1]
        X = frames.transpose((1,0)) # Convert to (T, 64)
        # Flatten spatial: reshape to (T, 64)
        X = frames.reshape(T, -1, order="F")  # X.shape = (T, 64)

        # --- PCA ---
        pca = PCA(n_components=components, svd_solver='full')  # Keep components
        X_pca = pca.fit_transform(X)
        X_reconstructed = pca.inverse_transform(X_pca)

        # Compute reconstruction error per channel (column)
        recon_error = np.mean((X - X_reconstructed) ** 2, axis=0)  # shape: (64,)

        # Threshold: mark bad channels
        threshold = np.percentile(recon_error, 100 - top_error)  # top x% error
        bad_channels = recon_error > threshold

        # Reshape for visualization
        recon_error_grid = recon_error.reshape(H, W, order="F")
        bad_mask = bad_channels.reshape(H, W, order="F")

        if verbose:
            # --- Plot ---
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(np.mean(frames, axis=-1).reshape(H,W), cmap='viridis')
            plt.title("Mean Frame")
            plt.colorbar()

            plt.subplot(1, 3, 2)
            plt.imshow(recon_error_grid, cmap='inferno')
            plt.title("PCA Reconstruction Error")
            plt.colorbar(label="MSE")

            plt.subplot(1, 3, 3)
            plt.imshow(bad_mask, cmap='gray')
            plt.title("Detected Bad Channels (PCA)")
            plt.colorbar(label="Bad=True")

            plt.tight_layout()
            plt.show()

            print("Bad channel indices:", np.argwhere(bad_mask))
        return np.argwhere(bad_mask), recon_error_grid

    def snr(self, frames, H=8, W=8, threshold=40, verbose=False):
        indices = []
        ratios = []
        for i in range(H):
            indices.append([])
            ratios.append([])
            for j in range(W):
                n = (i * W) + j
                ratio = _estimate_snr(frames[n])
                if ratio < threshold:
                    indices[-1].append(1.0)
                else:
                    indices[-1].append(0.0)
                ratios[-1].append(ratio)
        if verbose:
            print("Indices")
            print(indices)
            print("Ratios")
            print(ratios)
        return np.argwhere(np.array(indices)), np.array(ratios)
    
    def spatialGrubbsPixel(self, frames, H=8, W=8, alpha=0.01, t_window=1, verbose=False):
        neighbors_total = 8*(t_window + 1)
        G_crit = generate_g_crit(neighbors_total, alpha)
        if verbose:
            print(G_crit)
            print(frames.shape)
        channel_results = [spatial_grubbs_pixel_core(frames, i, j, G_crit, t_window) for i in range(H) for j in range(W)]
        #channel_results = joblib.Parallel(n_jobs=64)(joblib.delayed(spatial_grubbs_pixel_core)(data, i, j, G_crit) for i in range(H) for j in range(W))
        if verbose:
            print(channel_results)
        grubbs_bad_channels = np.array([[i,j] for i in range(H) for j in range(W) if channel_results[(i * W) + j][1]])
        return grubbs_bad_channels, channel_results