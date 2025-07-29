import numpy as np
from toksearch import Pipeline, MdsSignal
from toksearch_d3d import PtDataSignal


def create_pipeline(shots):

    pipe = Pipeline(shots)

    #chlist = np.arange(64) + 1
    chlist = np.arange(64) + 1

    #pipe.fetch("r", MdsSignal(r"\bes_r", "bes")) #good logic, bug due to MDSplus pathing on omega
    #pipe.fetch("z", MdsSignal(r"\bes_z", "bes"))


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

if __name__ == "__main__":

    shots = [190850,]

    pipe = create_pipeline(shots)

    result = pipe.compute_serial()[0]

    print(result)