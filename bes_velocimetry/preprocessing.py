import numpy as np
import argparse
import create_structure as cs
import bes_velocimetry.bes_filter as bes_filter
import scipy.interpolate as si
import imageio
import os

def generate_gif(shot,fnlist,output_fn1):
    # Combine into a GIF
    output_gif = './IMAGES/'+str(shot)+'/'+output_fn1+'.gif'
    frames = [imageio.v3.imread(img) for img in fnlist]
    imageio.mimsave(output_gif, frames, duration=0.1)  # Adjust duration (in seconds) as needed
    print(f"GIF saved as {output_fn1}")
    os.system('rm '+'./IMAGES/'+str(shot)+'/*png')  
    return

def main():
    parser = argparse.ArgumentParser(description='check bes') #might want to move this parser out of main, food for later.
    #parser.add_argument('--shot', help='shot number to get the data', type=int, required=True) #no need for shot specific params
    parser.add_argument('--input', help='Raw file to be preprocessed', type=str, default=None, required=True)
    parser.add_argument('--out', help="Path for output file", type=str, default=None, required=True)
    parser.add_argument('--bandwidth', help='Sets staggering for bandpass filtering in KHz', type=int, default=10)
    parser.add_argument('--ceiling', help='Highest frequency to examine for all bands in KHz', type=int, default=500)
    parser.add_argument('--floor', help='Lowest frequency to examine for all bands in KHz', type=int, default=10)
    parser.add_argument('--badchannels', help='List of channels to be ignored, should live in data fetcher eventually', default=np.array([7,40]))
    parser.add_argument('--makegif', help='Produces a gif of the filtered data', default=False)
    args = parser.parse_args()
    
    out_dir = args.out + "/"
    a = cs.from_h5file(args.input)
    shot = a.shot #cludge for output filename
    ch_missing = args.badchannels
    savepng = args.makegif
    #Calculate the cutoff arrays to be used for each file
    print((args.ceiling-args.floor)/args.bandwidth)
    temp_cutoff = np.array([args.floor, args.floor+args.bandwidth])
    cutoff= temp_cutoff


    data0 = a.fdata
    time0 = a.t_fdata[0]
    dt = (time0[-1]-time0[0])/time0.shape[0]
    nch = data0.shape[0]
    # frequency filter
    print('------filtering-------')
    data1 = np.zeros_like(data0)
    print('cut off f:',cutoff)
    for i in range(nch):
        data1[i,:] = bes_filter.bandpass(time0,data0[i,:],cutoff)

    interp_factor = 3
    time2 = np.arange(time0.min(),time0.max(),dt/interp_factor)
    for i in range(nch):
        f = si.interp1d(time0,data1[i,:],kind='linear',
                fill_value="extrapolate")
        if i ==0:
           data2 = np.zeros((data1.shape[0],time2.shape[0]))
        data2[i,:] = f(time2)
    # interpolate in RZ plane
    ch_width = 0.8
    ch_height = 1.1
    res = 10*8 #8*8 array changes to 80*80
    # Define the interpolation grid
    R0 = min(a.bes_r) - ch_width / 2.
    R1 = max(a.bes_r) + ch_width / 2.
    Z0 = min(a.bes_z) - ch_height / 2.
    Z1 = max(a.bes_z) + ch_height / 2.
    Ri, Zi = np.meshgrid(np.linspace(R0, R1, num=res+1), np.linspace(Z0, Z1, num=res))
    nframes = data2.shape[1]
    data3 = np.zeros((nframes,Ri.shape[0],Ri.shape[1]))
    print('------interpolation------')
    exit
    for i in range(nframes):
           #rbf = si.Rbf(a.bes_r[1,:],a.bes_z[1,:],data2[:,i],function='thin_plate')
           xtmp = np.delete(a.bes_r[1,:],ch_missing-1,axis=0)
           ytmp = np.delete(a.bes_z[1,:],ch_missing-1,axis=0)
           ztmp = np.delete(data2[:,i]  ,ch_missing-1,axis=0)
           rbf = si.Rbf(xtmp,ytmp,ztmp,function='thin_plate')
           data3[i,:,:] = rbf(Ri,Zi)
    # scaled to ... copied from IDL
    nc = 128
    sd = 5
    data4 = data3 - np.sum(data3)/data3.size
    rms = np.sqrt(np.sum(data4**2)/(data4.size-1))
    data4 = data4*(nc/2)/(sd*rms)+nc/2
    data4[data4<1] = 1
    data4[data4>nc-1] = nc-1
    fnlist = []

    print('SAVING HDF ... ')
    out = cs.data(' ')
    out.image_data  = data4
    out.xv = Ri
    out.yv = Zi
    out.time = time2
    out.xr =xr
    out.cutoff = cutoff
    out.shot = shot
    out.ch_missing = ch_missing
    save_h5.from_object(out, f'{out_dir}{processed}/{out_filename}')

    if savepng:
        print('SAVING PNG ...')
        for i in range(nframes):
            plt.figure(1);plt.clf()
            #plt.pcolormesh(data4[i,:,:],cmap='bwr',vmin=0,vmax=128)
            plt.contourf(Ri,-1*Zi,data4[i,:,:],cmap='bwr',vmin=0,vmax=128)
            fn = 'b'+str(int(xr[0]*1e3))+'.i'+str(i)+'.png'
            plt.title(str(shot)+'; i: '+str(i)+'; t:'+
                      str(np.round(time2[i],8)))
            plt.savefig('./IMAGES/'+str(shot)+'/'+fn,dpi=50)
            fnlist.append('./IMAGES/'+str(shot)+'/'+fn)
        generate_gif(shot,fnlist,output_fn1)
        for f in fnlist:
            os.remove(f)


if __name__ == '__main__':
    main()