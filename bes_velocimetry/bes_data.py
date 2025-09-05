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

class BES_data:
    def __init__(self):
        pass

    @classmethod
    def from_idlsave(cls, path):
        from scipy.io import readsav 
        data = readsav(path)
        o = cls.from_dict(data)
        return o
    
    @classmethod
    def from_h5file(cls, path):
        import h5py
        f = h5py.File(path,'r')
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
        f.close()
        return o
    
    @classmethod
    def from_dict(cls, dict):
        o = data('from_h5 ')
        keys = list(dict.keys())
        for i in range(len(keys)):
            o.__setattr__(keys[i],dict[keys[i]])
        return o
    
    @classmethod
    def from_object(cls, bes_data, path = './'):
        """
        save all variables in the sturcture to hdf5 file
        """
        f = h5py.File(path, "w")

        dim = len(dir(bes_data))
        for i in range(dim):
            if dir(bes_data)[i][0] != '_':
                f.create_dataset(dir(bes_data)[i],data = bes_data.__dict__[dir(bes_data)[i]])

        f.close()
        print(f'HDF FILE IS SAVED AS {path}')
        return