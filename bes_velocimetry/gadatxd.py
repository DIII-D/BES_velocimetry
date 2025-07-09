import MDSplus
import numpy as np
import time
import sys
#import MDS_SERVER
import sys
sys.path.append('/home/duxiaodi/xpsi/py_commons')
import create_structure as cs

def read(pn,shot,tree='d3d'):
    pointname = '\\'+pn
    print('*read in:  ', pointname)
    c = MDSplus.Connection('atlas.gat.com')
    c.openTree(tree,shot)
    try:
       d = c.get(pointname).data()
       t = c.get('DIM_OF('+pointname+')').data() 
    except:
       d = 0
       t = 0
    return np.asarray(t/1e3),np.asarray(d)

def ptdata(pn,shot):
    print('*read in:  ', pn)
    c = MDSplus.Connection('atlas.gat.com')
    # 2024Aug29, modified by Sean for efficiency improvement
    d = c.get('_data = '+'PTDATA("'+pn+'",'+str(shot)+')')
    t = c.get('DIM_OF(_data)')
#    d = c.get('PTDATA("'+pn+'",'+str(shot)+')')
#    t = c.get('DIM_OF('+'PTDATA("'+pn+'",'+str(shot)+')'+')')
    return np.asarray(t/1e3),np.asarray(d)

def multi(shot,sig,tree):
    a = cs.data(' ')
    for i in np.arange(len(sig)):
        print(tree[i])
        if tree[i] != 'ptdata':
           t,d = read(sig[i],shot,tree=tree[i])
        if tree[i] == 'ptdata':
           t,d = ptdata(sig[i],shot)
        a.__setattr__('t_'+sig[i],t) 
        a.__setattr__(sig[i],d)
    return a

def multi2(shot,sig,tree,t2):
    import findxr
    import scipy.interpolate
    import smooth
    xr = np.array([t2.min(),t2.max()])

    a = cs.data(' ')
    for i in np.arange(len(sig)):
        print(tree[i])
        if tree[i] != 'ptdata':
           t,d = read(sig[i],shot,tree=tree[i])
        if tree[i] == 'ptdata':
           t,d = ptdata(sig[i],shot)

        mask = ~np.isnan(d)
        t = t[mask]
        d = d[mask]
        try:  
           t0,d0 = findxr.main(t,smooth.smooth(d,10),xr)
           fd = scipy.interpolate.interp1d(t0,d0,fill_value="extrapolate")
           d2 = fd(t2)

           a.__setattr__(sig[i],d2)
        except:
           d2 = np.repeat(0,t2.shape[0])
           a.__setattr__(sig[i],d2)

    a.__setattr__('shot',shot)
    a.__setattr__('time',t2)
    return a


def nb(shot):
    sig = \
      ['PINJ',
       'PINJ_30R',
       'PINJ_30L',
       'PINJ_15R',
       'PINJ_15L',
       'PINJ_21R',
       'PINJ_21L',
       'PINJ_33R',
       'PINJ_33L' ]
    tree = np.repeat('d3d',9) 

    a =cs.data(' ')
    a = multi(shot,sig,tree)
    a.pinj_r = a.PINJ_30R+a.PINJ_15R+a.PINJ_33R
    a.pinj_l = a.PINJ_30L+a.PINJ_15L+a.PINJ_33L
    a.pinj_c = a.PINJ_21R+a.PINJ_21L

    return a

def nbvolt(shot):
    sig = \
      [
       'pcnbv30rt',
       'pcnbv30lt',
       'pcnbv15rt',
       'pcnbv15lt',
       'pcnbv21rt',
       'pcnbv21lt',
       'pcnbv33rt',
       'pcnbv33lt' ]
    tree = np.repeat('ptdata',9)

    a =cs.data(' ')
    a = multi(shot,sig,tree)
    a.pinj_r = a.pcnbv30rt+a.pcnbv15rt+a.pcnbv33rt
    a.pinj_l = a.pcnbv30lt+a.pcnbv15lt+a.pcnbv33lt
    a.pinj_c = a.pcnbv21lt+a.pcnbv21rt

    return a


def ece(shot):
    sig = []
    for i in range(31):
        sig.append('tecef'+str(i+1).zfill(2))

    for i in range(31):
        t,d = gadatxd.read(sig[i],shot)
        if i == 0:
           a = cs.data(' ')
           a.t = t
           a.d = np.zeros((31,d.shape[0]))
        a.d[i] = d

    a.shot = shot
    return a

def co2(shot,ch,xr,ptn='pl1'):
    tseg \
    = [-200,1017.8890490692138,2276.1808490692138,3534.4714490692136,4792.7626490692137,6051.0538490692143,7309.3450490692130,7496.5066490692125]
    xr = np.asarray(xr)
    tseg = np.asarray(tseg)/1e3
    loc = np.zeros(2,dtype='int')
    loc[0] = int(np.argmin(np.abs(xr[0]-tseg)))
    loc[1] = int(np.argmin(np.abs(xr[1]-tseg)))
    if tseg[loc[0]]>xr[0]:
       loc[0] = loc[0]-1
    if tseg[loc[1]]<xr[1]:
       loc[1] = loc[1]+1
    print(loc)

    ii = 0
    for i in np.arange(loc[0],loc[1]+1,1):
#        pointname = 'pl1'+ch+'_uf_'+str(int(i))
        pointname = ptn+ch+'_uf_'+str(int(i))
        t_tmp1,d_tmp1 = read(pointname,shot)
        if ii == 0:
           t_tmp = np.copy(t_tmp1)
           d_tmp = np.copy(d_tmp1)
           ii += 1
        else:
           t_tmp = np.append(t_tmp,t_tmp1)
           d_tmp = np.append(d_tmp,d_tmp1)
           ii += 1

    mask = np.where((t_tmp<xr[1])&(t_tmp>xr[0]))[0]
    return t_tmp[mask],d_tmp[mask]

def co2_pl_to_density(shot,ch,xr):
    tseg \
    = [-200,1017.8890490692138,2276.1808490692138,3534.4714490692136,4792.7626490692137,6051.0538490692143,7309.3450490692130,7496.5066490692125]
    xr = np.asarray(xr)
    tseg = np.asarray(tseg)/1e3
    loc = np.zeros(2,dtype='int')
    loc[0] = int(np.argmin(np.abs(xr[0]-tseg)))
    loc[1] = int(np.argmin(np.abs(xr[1]-tseg)))
    if tseg[loc[0]]>xr[0]:
       loc[0] = loc[0]-1
    if tseg[loc[1]]<xr[1]:
       loc[1] = loc[1]+1
    print(loc)

    ii = 0
    for i in np.arange(loc[0],loc[1]+1,1):
        pointname1 = 'pl1'+ch+'_uf_'+str(int(i))
        pointname2 = 'pl2'+ch+'_uf_'+str(int(i))
        t_tmp1,d_tmp1 = read(pointname1,shot)
        t_tmp2,d_tmp2 = read(pointname2,shot)
        print(ii)
        if ii == 0:
           t1 = np.copy(t_tmp1)
           d1 = np.copy(d_tmp1)
           t2 = np.copy(t_tmp2)
           d2 = np.copy(d_tmp2)
           ii += 1
        else:
           t1 = np.append(t1,t_tmp1)
           d1 = np.append(d1,d_tmp1)
           t2 = np.append(t2,t_tmp2)
           d2 = np.append(d2,d_tmp2)
           ii += 1
    
    lambda_co2 = 10.591
    if ch == 'r0':
       lambda_hene = 3.3922
    if ch == 'v1':
       lambda_hene = 0.63299
    if ch == 'v2':
       lambda_hene = 0.63297
    if ch == 'v3':
       lambda_hene = 0.633

    phi_vc = d1-d2*lambda_hene/lambda_co2
    mask = np.where((t1<xr[1])&(t1>xr[0]))[0]
     
    return t1[mask],phi_vc[mask]

    mask = np.where((t_tmp<xr[1])&(t_tmp>xr[0]))[0]
    return t_tmp[mask],d_tmp[mask]

def co2_pl_normalize(ch,shot):
    """
  pl1r0_uf_i is in radians but it includes vibration at low frequencies. If you are looking at higher frequencies, ~10kHz, that’s ok but if you want to normalize by equilibrium density, you need to divide by vibration compensated phase (that’s ~ nL). You can do that by:  Phi_VC = pl1r0s - pl2r0s * lambda_hene / lambda_co2
  lambda_co2 = 10.591
  Lambda_hene R0, V1, V2, V3 = 3.3922, 0.63299, 0.63297, 0.633
    """
    t_pl1s, pl1s = read('pl1'+ch+'s',shot)
    t_pl2s, pl2s = read('pl2'+ch+'s',shot)
    lambda_co2 = 10.591
    if ch == 'r0':
       lambda_hene = 3.3922
    if ch == 'v1':
       lambda_hene = 0.63299
    if ch == 'v2':
       lambda_hene = 0.63297
    if ch == 'v3':
       lambda_hene = 0.633


    phi_vc = pl1s - pl2s*lambda_hene/lambda_co2
     
    return t_pl1s,phi_vc


        
        

