import numpy as np
from scipy.signal import firwin, freqz, filtfilt
from scipy import interpolate

def bandpass(t,d,cutoff,numtaps=501,plot_ftf=False):
    """
    This function filters BES data using a non-causal FIR filter. The filter
    has linear phase response and is applied with a forwards-backwards
    algorithm so there is not net time shift of the data.

    Parameters
    ==========
    signals : xarray DataArray
        Signals to be filtered. Must have dimensions 'channel' and
        'time'. Must have attribute 'dt' giving sampling time in ms.
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
    filt_signals : xarray DataArray
        Bandpass filtered signals.
    """

    # Calculate FIR filter coefficients
    fs = 1 / (t[1]-t[0])/1e3 #kHz
    # bandpass filter; fs and cutoff frequency should have same unit
    b = firwin(numtaps, cutoff, pass_zero=False, fs=fs)

    # Plot filter transfer function
    if plot_ftf:
        w, h = freqz(b)
        freqs = w * fs / (2 * np.pi)  # convert units from rad/sample to Hz
        mag_dB = 20 * np.log10(np.absolute(h))
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
#    filt_signals = signals.copy()  # avoid modifying input signals
    filt_signals = filtfilt(b, 1, d)
#    filt_signals.settings.update({'cutoff_freqs': cutoff, 'numtaps': numtaps}) 

    return filt_signals
