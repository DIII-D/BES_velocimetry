import numpy as np
from scipy.signal import firwin, freqz, filtfilt
from scipy import interpolate

def bandpass(data, dt, cutoff, numtaps=501, plot_ftf=False):
    """
    This function filters BES data using a non-causal FIR filter. The filter
    has linear phase response and is applied with a forwards-backwards
    algorithm so there is not net time shift of the data.

    Parameters
    ==========
    data : ndarray (N_chan, N_time)
    cutoff : array of floats
        Filter cutoff frequencies in kHz.
    numtaps : int, optional
        Number of FIR filter taps to use. More taps improves sharpness of the
        transition from passband to stopband at the cost of increased
        computation time.
    plot_ftf : bool, optional
        Plots the filter's transfer function (magnitude and phase).

    Returns
    =======
    filt_data : ndarray
        Bandpass filtered  data.
    """

    # Calculate FIR filter coefficients
    fs = 1 / dt  # kHz
    if cutoff[0] == 0 or cutoff[0] == None:  # lowpass filter
        b = firwin(numtaps, cutoff[1:], pass_zero=True, fs=fs)
    elif cutoff[-1] == fs / 2.0 or cutoff[-1] == None:  # highpass filter
        b = firwin(numtaps, cutoff[:-1], pass_zero=False, fs=fs)
    else:  # bandpass filter
        b = firwin(numtaps, cutoff, pass_zero=False, fs=fs)

    # Plot filter transfer function
    if plot_ftf:
        w, h = freqz(b)
        freqs = w * fs / (2 * np.pi)  # convert units from rad/sample to Hz
        mag_dB = 20 * np.log10(np.abs(h))
        phase = np.unwrap(np.angle(h))

        fig, ax1 = plt.subplots()
        ax1.plot(freqs, mag_dB, 'b-')
        ax1.set_xlabel('Frequency (kHz)')
        ax1.set_ylabel('Amplitude (dB)', color='b')
        ax2 = ax1.twinx()
        ax2.plot(freqs, phase, 'r')
        ax2.set_ylabel('Phase (rad)', color='r')
        ax1.set_title('Filter transfer function')
        fig.tight_layout()

    # Filter signals
    filt_data = filtfilt(b, 1, data, axis=1)

    return filt_data


def apply_transfer_functions(data, dt):
    """
    Applies transfer function correction to input signals by Fourier
    transforming to the frequency domain, dividing by the transfer
    function, then inverse Fourier transforming back to time domain.

    data : ndarray
    dt : float
    """
    # Load transfer functions
    tf = np.loadtxt("133298_tf.csv", delimiter=",")
    tf_data = tf[1:, :]  # 0th row is freq
    tf_frequency = tf[0, :] / 1000.0  # Hz -> kHz
    # Apply to data
    nchannels = data.shape[0]
    length = data.shape[1]
    data = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(length, d=dt)  # kHz
    splines = interpolate.interp1d(tf_frequency, tf_data, kind='cubic', fill_value='extrapolate')
    data = np.fft.irfft(data / np.sqrt(np.abs(splines(freqs))), n=length)

    return data


def filter_nbi(data, sig_time, filter_ds, analysis_times=[0, 9500], keep_nans=False):
    """
    keeps data only at times when the viewed 150 beam is on
    and another one is off, the viewed beam is determined from
    mds parameter 'BES::TOP.BEAM'

    Parameters
    ==========
    data : ndarray
    sig_time : ndarray
        data timebase
    device : string
    shot: int
    analysis_times: list
        list with start and finish time
    keep_nans : bool
        If True, then the resulting data array has all the data points,
        but the data during vieved beam off (or both beams on) time is set to np.nan
        If keep_nans=False, then the data during vieved beam off (or both beams on) time
        is removed from the data array
    Returns
    =======
    data : ndarray
    """

    print('\nFiltering NBI modulation')
    # Check viewed beam (150-Left or 150-Right)
    # Get NBI info
    beam_index = filter_ds['bes_beam']['data']
    if beam_index == 0:  # 150-R beam
        viewed_beam = filter_ds['pinj_15r']['data']
        beam_time = filter_ds['pinj_15r']['times']
        odd_beam = filter_ds['pinj_15l']['data']
        print('BES focused on 150-RIGHT')
    elif beam_index == 1:  # 150-L beam
        viewed_beam = filter_ds['pinj_15l']['data']
        beam_time = filter_ds['pinj_15l']['times']
        odd_beam = filter_ds['pinj_15r']['data']
        print('BES focused on 150-LEFT')
    else:
        raise OMFITexception('Cannot determine which DIII-D beam is being viewed.')
    # Take beam data at analysis_times
    dt = beam_time[1] - beam_time[0]
    tmin, tmax = analysis_times
    mask = (beam_time > tmin) & (beam_time < tmax)
    beam_time = beam_time[mask]
    viewed_beam = viewed_beam[mask]
    odd_beam = odd_beam[mask]
    # Set up times when only viewed beam is on
    selected_times = beam_time[(viewed_beam > 1e4) & (odd_beam < 1e4)][1:-1]
    indices = np.where(selected_times[1:] - selected_times[0:-1] > 2 * dt)[0]
    indices_start = np.insert(indices + 1, 0, 0)
    indices_end = np.insert(indices, len(indices), len(selected_times) - 1)
    # Break time sequence into pairs [start_time, end_time]
    # Remove 2 ms at the beginning and 0.5 ms at the end of each NBI blip
    selected_times_filtered = [[selected_times[x] + 2, selected_times[y] - 0.5] for x, y in zip(indices_start, indices_end)]
    # Find indices of data points at times when only viewed beam is on
    time_indices = []
    for time_range in selected_times_filtered:
        tmin, tmax = time_range
        time_indices += np.where((sig_time > tmin) & (sig_time < tmax))[0].tolist()
    # Filter data using the found indices
    print(f'Detected {len(selected_times_filtered)} NBI blip(s)')
    # print('selected times after filtering NBI modulation: ', selected_times_filtered)
    # print('Total amount of data points: {}'.format(len(data.shape[1])))
    if keep_nans:
    	# Set data values during beam off (or both beams on) times to np.nan
        time_mask = np.zeros_like(sig_time, dtype=bool)
        time_mask[time_indices] = True
        data[:, ~time_mask] = np.nan
        print('Data during beam off times is set to NaN\n')
    else:
    	# Remove data values during beam off (or both beams on) times
        data = data[:, time_indices]
        print('Data during beam off times removed from the array\n')
    # print('Amount of data points after filtering NBI modulation: {}'.format(len(data.shape[1])))

    return data


def modified_zscore(data):
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    modified_z = 0.6745 * (data - median) / mad
    return modified_z


def find_bad_channels(data, time, threshold_low=0.005, threshold_high=0.1):
    '''
    The function works with NBI-filtered data where NBI-off times are set to nans
    It breaks data into time slices removing nans
    For bad channels check - simple logic - check the modified Z-score
    Returns list of data and time slices, and lists of bad channels for each time slice
    '''
    data_list = []
    time_list = []
    bad_channels_list = []
    # Check if data has nans first
    if not np.isnan(data).any():
        # If no nans found, don't slice the data
        data_list.append(data)
        time_list.append(time)
    else:
        # If nans found, slice data to remove all nans
        # Check for nan in the 1st channel
        mask = ~np.isnan(data[0, :])
        # Find start and end indices of non-nan data segments
        diff = np.diff(np.concatenate(([False], mask, [False])).astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        # Extract each continuous data segment
        for start, end in zip(starts, ends):
            data_list.append(data[:, start:end])
            time_list.append(time[start:end])
    #print(f'found {len(data_list)} data slice(s)')
    # For each data slice find bad channels
    for data_slice in data_list:
        std = np.nanstd(data_slice, axis=1)  # calculate std over the time axis
        score = np.abs(modified_zscore(np.log(1e3*std + 1e-3)))
        bad_channels = np.where(score > 2.5)[0] + 1
        #bad_channels = np.where((std < threshold_low)|(std > threshold_high))[0] + 1  # channel numbers start from 1
        bad_channels_list.append(bad_channels.tolist())

    return data_list, time_list, bad_channels_list
    
