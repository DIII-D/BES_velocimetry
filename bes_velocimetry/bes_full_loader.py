import numpy as np
import argparse
import bes_data
import MDSplus

'''
Tasks:
Sort out filtering bad channels (can be manual list for prototyping purposes)
'''

def _ptdata(c, pn,shot):
    print('*PTDATA read in:  ', pn)
    d = c.get('_data = '+'PTDATA("'+pn+'",'+str(shot)+')')
    t = c.get('DIM_OF(_data)')
    return np.asarray(t/1e3),np.asarray(d) #guessing this modification to t is modification to gadatxd for BES?

class BESRemoteFeed:
    host = None

    def __init__(self, host="atlas.gat.com"):
        self.host = host

    def readShot(self, shot):
        c = MDSplus.Connection(self.host)
        chlist = np.arange(64)+1
        print('shot: ' + str(shot))
        c.openTree('bes',shot) #This should be more specific
        bes_r = c.get('\\bes_r').data()
        bes_z = c.get('\\bes_z').data()

        s_list, ts_list = [], []
        f_list, tf_list = [], []
        for i_ch in chlist:
            #print(str(i_ch).zfill(2))
            t1s,d1s =  _ptdata(c,'BESSU'+str(i_ch).zfill(2),shot)
            t1f,d1f =  _ptdata(c,'BESFU'+str(i_ch).zfill(2),shot)
            s_list.append(d1s)
            ts_list.append(t1s)
            f_list.append(d1f)
            tf_list.append(t1f)
        sdata   = np.stack(s_list,   axis=0)  # shape (cdim, N)
        t_sdata = np.stack(ts_list,  axis=0)
        fdata   = np.stack(f_list,   axis=0)
        t_fdata = np.stack(tf_list,  axis=0)

        out = {}
        out["chlist"] = chlist
        out["shot"] = shot
        out["sdata"] = sdata
        out["t_sdata"] = t_sdata
        out["fdata"] = fdata
        out["t_fdata"] = t_fdata
        out["bes_r"] = bes_r
        out["bes_z"] = bes_z
        return bes_data.BES_data.from_dict(out)

if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser(description='Creates hdf5 file containing raw BES data.')
        parser.add_argument('--shot', help='shot number to get the data', type=int, required=True)
        #parser.add_argument('--ch', help='channel number to get the data', type=int, default=64) #should be an array eventually
        parser.add_argument('--good-channels', help='triggers logic to filter which channels to use', default=False, type=bool, required=False)
        parser.add_argument("--out", help="Path for output file", type=str, default=None, required=True)
        parser.add_argument("--host", help="Remote host", type=str, default="atlas.gat.com", required=True)
        args = parser.parse_args()
        shot = args.shot
        host = args.host
        out_dir = args.out + "/"
        feed = BESRemoteFeed(host)
        out = feed.readShot(shot)
        bes_data.BES_data.save_h5_from_object(out,path=out_dir + 'raw/bes.s'+str(shot)+'.h5')
    main()