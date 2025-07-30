import numpy as np
from toksearch import Pipeline, MdsSignal
from toksearch_d3d import PtDataSignal
import argparse

import bes_velocimetry.create_structure as cs #These should be removed
import bes_velocimetry.save_h5 as save_h5 #Once this is known to be sound/solid.


def raw_bes_pipeline(shots):
    pipe = Pipeline(shots)
    chlist = np.arange(64) + 1
    slow_signals_dict = {}
    fast_signals_dict = {}
    for i_ch in chlist:
        su_ptname = f"BESSU{i_ch:02d}"
        fu_ptname = f"BESFU{i_ch:02d}"
        slow_signals_dict[su_ptname] = PtDataSignal(su_ptname)
        fast_signals_dict[fu_ptname] = PtDataSignal(fu_ptname)
    pipe.fetch_dataset("slow_ds", slow_signals_dict)
    pipe.fetch_dataset("fast_ds", fast_signals_dict)

    return pipe

def preprocessing_pipeline(shots):
    pipe = Pipeline(shots)

    pipe.fetch("bes_r", MdsSignal(r"\bes_r", "bes", location="remote://atlas.gat.com")) #good logic, bug due to MDSplus pathing on omega
    pipe.fetch("bes_z", MdsSignal(r"\bes_z", "bes", location="remote://atlas.gat.com"))
    pipe.fetch("bes_beam", MdsSignal(r"\bes_beam", "bes", location="remote://atlas.gat.com", dims=[]))
    pipe.fetch("pinj_15l", MdsSignal(r"\pinj_15l", "nb", location="remote://atlas.gat.com"))
    pipe.fetch("pinj_15r", MdsSignal(r"\pinj_15r", "nb", location="remote://atlas.gat.com"))

    return pipe   

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Creates hdf5 file using raw BES data from Toksearch.')
    parser.add_argument('--shot', help='shot number to get the data', type=int, default=192095) #required=True)
    parser.add_argument('--good-channels', help='triggers logic to filter which channels to use', default=False, type=bool, required=False)
    parser.add_argument("--out", help="Path for output file", type=str, default=None, required=True)
    args = parser.parse_args()
    out_dir = args.out + "/"

    #raw_bes_ds = raw_bes_pipeline([args.shot]).compute_serial()[0]
    filter_ds = preprocessing_pipeline([args.shot]).compute_serial()[0]

    #print(raw_bes_ds)
    print(filter_ds)

    #All logic from here on will need some kind of modification either to make usage of the xarray data
    #(which would make code using MDSplus calls directly incompatible) or to extract and feed it into existing code.
    '''
    out = cs.data(' ')
    out.chlist = chlist
    out.shot = shot
    out.sdata = sdata
    out.t_sdata = t_sdata
    out.fdata = fdata
    out.t_fdata = t_fdata
    out.bes_r = bes_r
    out.bes_z = bes_z
    save_h5.from_object(out,path=out_dir+'raw/bes.s'+str(shot)+'.h5')
    '''