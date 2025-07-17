import sys
import os
import numpy as np
import argparse
import bes_velocimetry.create_structure as cs
import bes_velocimetry.save_h5 as save_h5
import MDSplus

'''
Tasks:
Rebake main array aggregation logic in for loop (don't make zero array and then stick in more stuff)
Sort out filtering bad channels (can be manual list for prototyping purposes)
Reimplement the masking logic (don't remember what it does, have resolved before)
'''

def ptdata(c, pn,shot):
    print('*PTDATA read in:  ', pn)
    d = c.get('_data = '+'PTDATA("'+pn+'",'+str(shot)+')')
    t = c.get('DIM_OF(_data)')
    return np.asarray(t/1e3),np.asarray(d) #guessing this modification to t is modification to gadatxd for BES?

def main():
    parser = argparse.ArgumentParser(description='check bes')
    parser.add_argument('--shot', help='shot number to get the data', type=int, required=True)
    #parser.add_argument('--ch', help='channel number to get the data', type=int, default=64) #should be an array eventually
    parser.add_argument("--out", help="Path for output file", type=str, default=None, required=True)
    args = parser.parse_args()
    shot = args.shot
    out_dir = args.out + "/"
    c = MDSplus.Connection('atlas.gat.com')
    chlist = np.arange(64)+1
    print('shot: ' + str(shot))
    c.openTree('bes',shot) #This should be more specific
    bes_r = c.get('\\bes_r').data()
    bes_z = c.get('\\bes_z').data()

    s_list, ts_list = [], []
    f_list, tf_list = [], []
    for i_ch in chlist:
        #print(str(i_ch).zfill(2))
        t1s,d1s =  ptdata(c,'BESSU'+str(i_ch).zfill(2),shot)
        t1f,d1f =  ptdata(c,'BESFU'+str(i_ch).zfill(2),shot)
        s_list.append(d1s)
        ts_list.append(t1s)
        f_list.append(d1f)
        tf_list.append(t1f)
    sdata   = np.stack(s_list,   axis=0)  # shape (cdim, N)
    t_sdata = np.stack(ts_list,  axis=0)
    fdata   = np.stack(f_list,   axis=0)
    t_fdata = np.stack(tf_list,  axis=0)

    out = cs.data(' ')
    out.chlist = chlist
    out.shot = shot
    out.sdata = sdata
    out.t_sdata = t_sdata
    out.fdata = fdata
    out.t_fdata = t_fdata
    out.bes_r = bes_r
    out.bes_z = bes_z
    save_h5.from_object(out,path='./raw/bes.s'+str(shot)+'.h5')



    #


if __name__ == '__main__':
    main()