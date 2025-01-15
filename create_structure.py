import h5py

class data(object):
    """
    Attributes:
    """
    def __init__(self,notes):
        self.notes = notes

class note(object):
    """
    Attributes:
    """
    def __init__(self,notes):
        self.notes = notes

def from_dict(f):
    o = data('from_h5 ')
    keys = list(f.keys())
    for i in range(len(keys)):
        o.__setattr__(keys[i],f[keys[i]])
    return o

def from_h5file(fn):
    import h5py
    f = h5py.File(fn,'r')
    o = data('from_h5 ')
    keys = f.keys()
    for i in range(len(keys)):
        try:
    #      o.__setattr__(keys[i],f[keys[i]])
           o.__setattr__(keys[i],f[keys[i]][()])
        except:
           keys = list(keys)
    #       o.__setattr__(keys[i],f[keys[i]])
           o.__setattr__(keys[i],f[keys[i]][()])
    f.close
    return o

def from_h5csr(fn):
    from scipy import sparse
    f = h5py.File(fn,'r')
    g2 = f['Mcsr']
    M1 = sparse.csr_matrix((g2['data'][:],g2['indices'][:],
    g2['indptr'][:]), g2.attrs['shape'])
    f.close()
    return M1

def from_idlsave(fn):
    from scipy.io import readsav 
    data = readsav(fn)
    o = from_dict(data)
    return o
     
