import numpy as np
import h5py

def publication(a,path,citation,data_provider,doc_number,figure_number,):
    """
    save all variables in the sturcture to hdf5 file
    main(a,path,citation,data_provider,doc_number,figure_number,):
    """
    f = h5py.File(path, "w")
    f.create_dataset('citation',data = citation)
    f.create_dataset('data_provider',data = data_provider)
    f.create_dataset('doc_number',data = doc_number)
    f.create_dataset('figure_number',data = figure_number)

    g = f.create_group('DATA')
    dim = len(dir(a))
    for i in range(dim):
        if dir(a)[i][0] != '_':
           g.create_dataset(dir(a)[i],data = a.__dict__[dir(a)[i]])

    f.close()
    print('HDF FILE IS SAVED AS ' + path)
    return


def from_object(a,path = './'):
    """
    save all variables in the sturcture to hdf5 file
    """
    f = h5py.File(path, "w")

    dim = len(dir(a))
    for i in range(dim):
        if dir(a)[i][0] != '_':
           f.create_dataset(dir(a)[i],data = a.__dict__[dir(a)[i]])

    f.close()
    print(f'HDF FILE IS SAVED AS {path}')
    return


def from_idlsave(fn,path='./'):
    import scipy.io
    a = scipy.io.readsav(fn)

    f = h5py.File(path, "w")
    dim = len(a.keys())
    tag_names = a.keys()
    for i in range(dim):
        f.create_dataset(tag_names[i],data = a[tag_names[i]])
    f.close()
    print('HDF FILE IS SAVED AS ' + path)
    return


def from_dictionary(a,path = './'):
    f = h5py.File(path, "w")

    dim = len(a.keys())
    tag_names = a.keys()
    for i in range(dim):
        f.create_dataset(tag_names[i],data = a[tag_names[i]])
    f.close()
    print('HDF FILE IS SAVED AS ' + path)


    return

def from_csr(M,path='./'):
    f = h5py.File(path,'w')
    g = f.create_group('Mcsr')
    g.create_dataset('data',data=M.data)
    g.create_dataset('indptr',data=M.indptr)
    g.create_dataset('indices',data=M.indices)
    g.attrs['shape'] = M.shape
    f.close()
    print('HDF FILE IS SAVED AS ' + path)

    return

