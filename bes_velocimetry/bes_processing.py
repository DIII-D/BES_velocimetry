from channel_anomaly import ChannelAnomaly
import numpy as np
import scipy
import bes_data
from scipy.signal import firwin, freqz, filtfilt
from scipy import interpolate

def _bandpass(data, dt, cutoff, numtaps=501, plot_ftf=False):
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


def _apply_transfer_functions(data, dt):
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
    #print(tf_frequency.shape, tf_data.shape)
    splines = interpolate.CubicSpline(tf_frequency, tf_data, axis=1, extrapolate=True)
    data = np.fft.irfft(data / np.sqrt(np.abs(splines(freqs))), n=length)

    return data


def _filter_nbi(data, sig_time, filter_ds, analysis_times=[0, 9500]):
    """
    Slice the data based on when the viewed 150 beam is on
    and another 150 beam is off. The viewed beam is determined from
    mds parameter 'BES::TOP.BEAM'

    Parameters
    ==========
    data : ndarray
    sig_time : ndarray
        data timebase
    analysis_times: list
        list with start and finish time
    Returns
    =======
    data_list : list
        list of np arrays with data
    time_list : list
        list of np arrays with time
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
        print('Cannot determine which DIII-D beam is being viewed.')
    # Take beam data at analysis_times
    dt = beam_time[1] - beam_time[0]
    tmin, tmax = analysis_times
    mask = (beam_time > tmin) & (beam_time < tmax)
    beam_time = beam_time[mask]
    viewed_beam = viewed_beam[mask]
    odd_beam = odd_beam[mask]
    # Set up times when only viewed beam is on
    selected_times = beam_time[(viewed_beam > 1e4) & (odd_beam < 1e4)][1:-1]
    # Find times indices when only viewed beam is on
    indices = np.where(selected_times[1:] - selected_times[0:-1] > 2 * dt)[0]
    indices_start = np.insert(indices + 1, 0, 0)
    indices_end = np.insert(indices, len(indices), len(selected_times) - 1)
    # Slice data and time based on indices and put slices into lists
    data_list = []
    time_list = []
    for i_start, i_stop in zip(indices_start, indices_end):
        # Remove 2 ms at the beginning and 0.5 ms at the end of each NBI blip
        tmin, tmax = selected_times[i_start] + 2, selected_times[i_stop] - 0.5
        time_indices = np.where((sig_time > tmin) & (sig_time < tmax))[0]
        time_list.append(sig_time[time_indices])
        data_list.append(data[:, time_indices])
    print(f'Detected {len(data_list)} NBI blip(s)')

    return data_list, time_list

def _modified_zscore(data):
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    modified_z = 0.6745 * (data - median) / mad
    return modified_z

# Alternate find channels function. May be used later.
def _find_bad_channels(data):
    '''
    The function works with a data time slice.
    For bad channels check - simple logic - check the modified Z-score
    Returns a list of bad channel indices (starting from 0)
    '''
    std = np.nanstd(data, axis=1)  # calculate std over the time axis
    score = np.abs(_modified_zscore(np.log(1e3*std + 1e-3)))
    bad_channels = np.where(score > 2.5)[0]  # these are channel indices, start from 0
    
    return bad_channels.tolist()


def _filter_bes(bes_ds, filter_ds, cutoff_freqs, analysis_times):
    '''
    The function takes xarray dataset with BES data, applies trasfer fucntions and filteing,
    and slices data based on NBI timing
    
    Returns 
    data_list : list
        list with data np.arrays, 
    time_list : list 
        list with time np.arrays 
    '''
    # Put BES-fast data into np.array
    bes_fast = np.array([bes_ds['fast_ds'][var] for var in bes_ds['fast_ds'].data_vars])  # shape (n_chan, n_time)
    # Get the time base
    bes_fast_time = bes_ds['fast_ds']['times'].data
    nt = bes_fast_time.shape[0]
    dt = bes_fast_time[1] - bes_fast_time[0]
    # Apply transfer functions first
    data_filtered = _apply_transfer_functions(bes_fast, dt)
    # Bandpass filter
    data_filtered = _bandpass(data_filtered, dt, cutoff=cutoff_freqs, numtaps=501, plot_ftf=False)
    # NBI filter
    data_list, time_list = _filter_nbi(data_filtered, bes_fast_time, filter_ds,
                                      analysis_times=analysis_times)  
   
    return data_list, time_list


class BES_Processing:
    anomaly_analysis = None
    H = 8
    W = 8

    def __init__(self):
        self.anomaly_analysis = ChannelAnomaly()

    # Frames and tframes are of shape (64, T)
    def filterTimeSeries(self, frames, tframes, xr):
        mask = np.where((tframes[0,:]<=xr[1])&(tframes[0,:]>=xr[0]))[0]
        frames_filtered = frames[:,mask]
        tframes_filtered = tframes[:, mask]
        return frames_filtered, tframes_filtered
    
    def upsampleTimeSeries(self, frames, tframes, interp_factor=3):
        time0 = tframes[0]
        nch = self.H * self.W
        dt = (time0[-1]-time0[0])/time0.shape[0]
        time1 = np.arange(time0.min(),time0.max(),dt/interp_factor)
        result_frames = np.zeros((nch,time1.shape[0]))
        result_tframes = np.zeros((nch, time1.shape[0]))
        for i in range(nch):
            f = scipy.interpolate.interp1d(time0,frames[i,:],kind='linear',
                    fill_value="extrapolate")
            result_frames[i,:] = f(time1)
            result_tframes[i,:] = time1
        return result_frames, result_tframes
    
    def directChannelWhitelist(self, blocked_channels):
        selected_channels_direct = np.array([(i * self.W) + j for i in range(self.H) for j in range(self.W)])
        for channel in blocked_channels:
            index = (channel[0] * self.W) + channel[1]
            selected_channels_direct = np.delete(selected_channels_direct, np.where(selected_channels_direct == index))
        return selected_channels_direct
    
    def filterChannels(self, frames, bes_r, bes_z, channel_whitelist):
        frames = frames[channel_whitelist]
        bes_r = bes_r[1,channel_whitelist]
        bes_z = bes_z[1,channel_whitelist]
        return frames, bes_r, bes_z

    def upsampleChannels(self, frames, bes_r, bes_z, upsample_factor=10, ch_width=0.8, ch_height=1.1):
        # interpolate in RZ plane
        res_w = upsample_factor*self.W
        res_h = upsample_factor*self.H
        # Define the interpolation grid
        R0 = min(bes_r) - ch_width / 2.
        R1 = max(bes_r) + ch_width / 2.
        Z0 = min(bes_z) - ch_height / 2.
        Z1 = max(bes_z) + ch_height / 2.
        Ri, Zi = np.meshgrid(np.linspace(R0, R1, num=res_w+1), np.linspace(Z0, Z1, num=res_h))
        nframes = frames.shape[1]
        result_frames = np.zeros((nframes,Ri.shape[0],Ri.shape[1]))
        print('*-----interpolation-----*')
        for i in range(nframes):
            ztmp = frames[:, i]
            rbf = scipy.interpolate.Rbf(bes_r,bes_z,ztmp,function='thin_plate')
            result_frames[i,:,:] = rbf(Ri,Zi)
        return result_frames, Ri, Zi

    def suggestBadChannels(self, frames, tframes, verbose=False):
        # SNR analysis
        sdata_bad_channels_snr, sdata_channel_quality_snr = self.anomaly_analysis.snr(frames, threshold=45, verbose=verbose)
        # Grubbs analysis
        sdata_bad_channels_grubbs, sdata_channel_quality_grubbs = self.anomaly_analysis.spatialGrubbsPixel(frames, alpha=0.01, t_window=2, verbose=verbose)
        # PCA analysis
        sdata_bad_channels_pca, sdata_channel_quality_pca = self.anomaly_analysis.pca(frames, components=64, top_error=1, verbose=verbose)
        # Isolation Forest Analysis
        sdata_bad_channels_isolation, sdata_channel_quality_isolation = self.anomaly_analysis.isolationForest(frames, contamination=0.025, estimators=1000, verbose=verbose)
        # Aggregate flagged channels
        total_channels = np.zeros((self.H,self.W))
        bad_channel_detections = [sdata_bad_channels_snr, sdata_bad_channels_grubbs, sdata_bad_channels_pca, sdata_bad_channels_isolation]
        for bad_channel_list in bad_channel_detections:
            for detection in bad_channel_list:
                total_channels[detection[0], detection[1]] += 1
        total_bad_channels = np.argwhere(total_channels >= 1)
        return total_bad_channels
    
    def bandpass(self, data, dt, cutoff=None):
        nch = self.H * self.W
        data_new = np.zeros_like(data)
        if cutoff is None:
            cutoff = np.array([30, 50]) # I assume it's kHZ, but it might need to be fixed
        for i in range(nch):
            data_new[i,:] = _bandpass(data[i,:],cutoff)
        return data_new
    
    def scaleCount(self, frames, nc=128, sd=5):
        data_new = frames - np.sum(frames)/frames.size
        rms = np.sqrt(np.sum(data_new**2)/(data_new.size-1))
        data_new = data_new*(nc/2)/(sd*rms)+nc/2
        data_new[data_new<1] = 1
        data_new[data_new>nc-1] = nc-1
        '''
        for i in range(nframes):
            plt.figure(1);plt.clf()
            #plt.pcolormesh(data4[i,:,:],cmap='bwr',vmin=0,vmax=128)
            plt.contourf(Ri,Zi,data4[i,:,:],cmap='bwr',vmin=0,vmax=128)
            fn = 'b'+str(int(xr[0]*1e3))+'.i'+str(i)+'.png'
            plt.title(str(shot)+'; i: '+str(i)+'; t:'+str(np.round(time2[i],8)))
            plt.savefig('./'+str(shot)+'/'+fn,dpi=50)
            fnlist.append('./'+str(shot)+'/'+fn)
        '''
        return data_new

    def rawToPostProcessed(self, a, xr, cutoff=None):
        frames = a.fdata
        tframes = a.t_fdata
        bes_r = a.bes_r
        bes_z = a.bes_z
        shot = a.shot
        frames, tframes = self.filterTimeSeries(frames, tframes, xr)
        time_sequence = tframes[0,:]
        dt = (time_sequence[-1]-time_sequence[0])/tframes.shape[0]
        frames = self.bandpass(frames, dt, cutoff)
        frames, tframes = self.upsampleTimeSeries(frames, tframes)
        bad_channels = self.suggestBadChannels(frames, tframes)
        channel_whitelist = self.directChannelWhitelist(bad_channels)
        frames, bes_r, bes_z = self.filterChannels(frames, bes_r, bes_z, channel_whitelist)
        frames, Ri, Zi = self.upsampleChannels(frames, bes_r, bes_z)
        frames = self.scaleCount(frames)
        out = bes_data.data(' ')
        out.image_data  = frames
        out.xv = Ri
        out.yv = Zi
        out.time = tframes
        out.xr = xr
        out.cutoff = cutoff
        out.shot = shot
        out.ch_missing = bad_channels
        return out