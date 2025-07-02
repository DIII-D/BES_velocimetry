import sys
import os
import numpy as np
import gadatxd
import create_structure as cs
import save_h5
import sys
import scipy.signal as ss
import bes_percentage as bp
import concurrent.futures
import argparse


def process_channel(ich, shot, chlist, fr, xr, t_pinj, pinj, args):
    ch1 = "BESFU" + str(chlist[ich]).zfill(2)
    t1, d1 = bp.read_norm(shot, ch1, xr)
    b = cs.data(" ")
    b.t1 = t1
    b.d1 = d1
    b.ch1 = ch1
    b.t_pinj = t_pinj
    b.pinj = pinj
    fn = "./data/bes" + str(shot) + "_ch" + str(ch1) + ".h5"
    save_h5.from_object(b, fn)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="check bes")
    parser.add_argument("--shot", help="shot number to get the data", type=int)
    parser.add_argument("--ch", help="channel number to get the data", type=int)
    args = parser.parse_args()
    shot = args.shot
    print("shot: " + str(shot))

    if args.ch == 0:
        chlist = np.arange(1, 65, 1)
    else:
        chlist = np.array([args.ch])

    xr = np.array([0.3, 2.3])
    cdim = chlist.shape[0]

    fr = [10, 150]
    directory = "./data/" + str(shot)
    if not os.path.exists(directory):
        os.makedirs(directory)

    t_pinj, pinj = gadatxd.read("pinj", shot, tree="d3d")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_channel, ich, shot, chlist, fr, xr, t_pinj, pinj, args
            )
            for ich in np.arange(cdim)
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Generated an exception: {exc}")
