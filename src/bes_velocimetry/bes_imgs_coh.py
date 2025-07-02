import bes_velocimetry.create_structure as cs
import bes_velocimetry.save_h5 as save_h5
import numpy as np
try:
    import cupyx.scipy.interpolate as si
except
import scipy.interpolate as si
import matplotlib.pylab as plt
import bes_velocimetry.bes_filter as bes_filter
import imageio


# List of PNG file paths
# images = ['image1.png', 'image2.png', 'image3.png']
def generate_gif(fnlist, output_fn1):
    # Combine into a GIF
    output_gif = "./" + str(shot) + "/" + output_fn1 + ".gif"
    frames = [imageio.v3.imread(img) for img in fnlist]
    imageio.mimsave(
        output_gif, frames, duration=0.1
    )  # Adjust duration (in seconds) as needed
    print(f"GIF saved as {output_gif}")
    return


def check():
    # plt.subplot(311)
    # plt.plot(t, signal1, label="Original Signal 1")
    # plt.plot(t, signal2, label="Original Signal 2", alpha=0.7)
    # plt.subplot(312)
    # plt.plot(f, Cxy, label="Coherence")
    # plt.legend()
    # plt.subplot(313)
    # plt.plot(t, filtered_signal1, label="Filtered Signal 1 (Coherent Part)")
    # plt.plot(
    #     t,
    #     filtered_signal2,
    #     label="Filtered Signal 2 (Coherent Part)",
    #     alpha=0.7,
    #     color="red",
    # )
    # plt.legend()
    return


if __name__ == "__main__":
    shot = 199968
    # shot = 199972
    # shot = 199975
    # shot = 200729
    if shot == 190850:
        xr = np.array([0.521, 0.523])
    if shot == 199968:
        # xr = np.array([0.655,0.657]);cutoff = np.array([31,64])
        xr = np.array([0.631, 0.634])
        cutoff = np.array([25, 55])
        ch_missing = np.array([5, 13, 33, 41, 43, 51])
    if shot == 199972:
        xr = np.array([0.32, 0.322])
        ch_missing = np.array([5, 13, 33, 41, 43, 51])
        cutoff = np.array([25, 125])
    if shot == 199975:
        xr = np.array([0.32, 0.322])
        ch_missing = np.array([5, 13, 33, 41, 43, 51])
        # cutoff = np.array([25,50])
        # cutoff = np.array([25,65])
        cutoff = np.array([75, 100])
        xr = np.array([0.32, 0.36])
        xr = np.array([0.36, 0.365])
        xr = np.array([0.4, 0.402])
    if shot == 200729:
        # case 1
        xr = np.array([0.5, 0.502])
        cutoff = np.array([100, 125])
        # case 2
        xr = np.array([0.7, 0.702])
        # all taes
        cutoff = np.array([100, 150])
        # tae1
        cutoff = np.array([97.2, 108.5])
        # tae2
        cutoff = np.array([108.5, 115.5])
        # tae3
        cutoff = np.array([113.8, 124.5])
        # tae4
        cutoff = np.array([125.9, 136.5])
        # tae5
        # cutoff = np.array([136.5,147])
        ch_missing = np.array([2, 5, 10, 13])

    fn_rawdata = "bes.s" + str(shot) + ".h5"
    output_fn1 = (
        "imgs." + str(int(xr[0] * 1e3)) + "." + str(int(xr[1] * 1e3)) + "." + fn_rawdata
    )
    print("input: " + fn_rawdata)
    print("output: " + output_fn1)
    path = "/home/duxiaodi/xpsi/velocimetry/python/"
    a = cs.from_h5file(path + fn_rawdata)

    mask = np.where((a.t_fdata[0, :] <= xr[1]) & (a.t_fdata[0, :] >= xr[0]))[0]
    data0 = a.fdata[:, mask]
    time0 = a.t_fdata[0, mask]
    dt = (time0[-1] - time0[0]) / time0.shape[0]
    nch = data0.shape[0]

    # frequency filter
    print("*-----filtering-----*")
    data1 = np.zeros_like(data0)
    print("cut off f:", cutoff)
    print("time window:", xr)
    for i in range(nch):
        data1[i, :] = bes_filter.bandpass(time0, data0[i, :], cutoff)

    #   signal1 = np.copy(data1[52,:])
    #   for i in range(nch):
    #       signal2 = np.copy(data1[i,:])
    #       t = np.copy(time0)
    #       fs = 1/(t[1]-t[0])
    #
    #       f, Cxy = coherence(signal1, signal2, fs=fs, nperseg=512)
    #       # Compute power spectral density for signal1 and signal2
    ##       f, Pxx = welch(signal1, fs=fs, nperseg=512)
    ##       f, Pyy = welch(signal2, fs=fs, nperseg=512)
    #
    #       # Coherence-weighted filter in the frequency domain
    #       H = np.sqrt(Cxy)  # Coherence weighting
    #       #filtered_signal1 = fftconvolve(signal1, H, mode='same')
    #       filtered_signal2 = fftconvolve(signal2, H, mode='same')
    #       data1[i,:] = np.copy(filtered_signal2)

    #   check()
    #   plt.tight_layout()
    #   plt.tight_layout()
    #   plt.show()

    # interpolate the data in time axis for a few times, interp_factor
    interp_factor = 3
    time2 = np.arange(time0.min(), time0.max(), dt / interp_factor)
    for i in range(nch):
        f = si.interp1d(time0, data1[i, :], kind="linear", fill_value="extrapolate")
        if i == 0:
            data2 = np.zeros((data1.shape[0], time2.shape[0]))
        data2[i, :] = f(time2)

    # interpolate in RZ plane
    ch_width = 0.8
    ch_height = 1.1
    res = 10 * 8  # 8*8 array changes to 80*80
    # Define the interpolation grid
    R0 = min(a.bes_r[1]) - ch_width / 2.0
    R1 = max(a.bes_r[1]) + ch_width / 2.0
    Z0 = min(a.bes_z[1]) - ch_height / 2.0
    Z1 = max(a.bes_z[1]) + ch_height / 2.0
    Ri, Zi = np.meshgrid(np.linspace(R0, R1, num=res + 1), np.linspace(Z0, Z1, num=res))

    nframes = data2.shape[1]
    data3 = np.zeros((nframes, Ri.shape[0], Ri.shape[1]))
    print("*-----interpolation-----*")
    for i in range(nframes):
        # rbf = si.Rbf(a.bes_r[1,:],a.bes_z[1,:],data2[:,i],function='thin_plate')
        xtmp = np.delete(a.bes_r[1, :], ch_missing - 1, axis=0)
        ytmp = np.delete(a.bes_z[1, :], ch_missing - 1, axis=0)
        ztmp = np.delete(data2[:, i], ch_missing - 1, axis=0)
        rbf = si.Rbf(xtmp, ytmp, ztmp, function="thin_plate")
        data3[i, :, :] = rbf(Ri, Zi)

    # scaled to ... copied from IDL
    nc = 128
    sd = 5
    data4 = data3 - np.sum(data3) / data3.size
    rms = np.sqrt(np.sum(data4**2) / (data4.size - 1))
    data4 = data4 * (nc / 2) / (sd * rms) + nc / 2
    data4[data4 < 1] = 1
    data4[data4 > nc - 1] = nc - 1
    fnlist = []
    for i in range(nframes):
        plt.figure(1)
        plt.clf()
        # plt.pcolormesh(data4[i,:,:],cmap='bwr',vmin=0,vmax=128)
        plt.contourf(Ri, Zi, data4[i, :, :], cmap="bwr", vmin=0, vmax=128)
        fn = "b" + str(int(xr[0] * 1e3)) + ".i" + str(i) + ".png"
        plt.title(str(shot) + "; i: " + str(i) + "; t:" + str(np.round(time2[i], 8)))
        plt.savefig("./" + str(shot) + "/" + fn, dpi=50)
        fnlist.append("./" + str(shot) + "/" + fn)

    out = cs.data(" ")
    out.image_data = data4
    out.xv = Ri
    out.yv = Zi
    out.time = time2
    out.xr = xr
    out.cutoff = cutoff
    out.shot = shot
    out.ch_missing = ch_missing
    save_h5.from_object(out, output_fn1)
    generate_gif(fnlist, output_fn1)


# plot_eddy()
