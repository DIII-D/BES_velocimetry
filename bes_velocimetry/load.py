import sys
import os
sys.path.append('/home/duxiaodi/xpsi/py_commons')
import gadatxd
import create_structure as cs
import save_h5
import numpy as np

shot = 190850
shot = 199968
shot = 199972
shot = 199975
shot = 200729
chlist = np.arange(64)+1

a = cs.data(' ')
cdim = chlist.shape[0]
for ich in np.arange(cdim):
    ch = str(chlist[ich]).zfill(2)
    print(ch)
    t1s,d1s =  gadatxd.ptdata('BESSU'+ch[-2:],shot)
    t1f,d1f =  gadatxd.ptdata('BESFU'+ch[-2:],shot)
    if ich == 0:
       sdata = np.zeros((cdim,d1s.shape[0]))
       t_sdata = np.zeros((cdim,t1s.shape[0]))
       fdata = np.zeros((cdim,d1f.shape[0]))
       t_fdata = np.zeros((cdim,t1f.shape[0]))
    sdata[ich,:] = np.copy(d1s)
    t_sdata[ich,:] = np.copy(t1s)
    fdata[ich,:] = np.copy(d1f)
    t_fdata[ich,:] = np.copy(t1f)

bes_r = gadatxd.read('bes_r',shot)
bes_z = gadatxd.read('bes_z',shot)

a.chlist = chlist
a.shot = shot
a.sdata = sdata
a.t_sdata = t_sdata
a.fdata = fdata
a.t_fdata = t_fdata
a.bes_r = bes_r
a.bes_z = bes_z
save_h5.from_object(a,path='./bes.s'+str(shot)+'.h5')




