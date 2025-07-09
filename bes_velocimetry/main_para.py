import bes_velocimetry.create_structure as cs
import bes_velocimetry.save_h5 as save_h5
import numpy as np
import bes_velocimetry.odp_idl as odp_idl
import argparse
from multiprocessing import Pool
import sys
from pathlib import Path
import os


def process_imageset(args2):
    """
    Function to process a single imageset in parallel.
    """
    # imageset,5 7,11,9,9
    imageset, nsteps, sm_param, m_frame, mx, my = args2
    print(imageset, nsteps, sm_param, m_frame, mx, my)
    return odp_idl.ODP(
        imageset, nsteps=nsteps, sm_param=sm_param, m_frame=m_frame, mx=mx, my=my
    )


def main():
    parser = argparse.ArgumentParser(description="check bes")
    parser.add_argument(
        "--fn", help="hdf5 file name as the input", type=str, required=True
    )
    parser.add_argument("--cores", help="number of cores, runs 20x faster on NERSC when set to 256", type=int, default=10)
    parser.add_argument("--nsteps", help="number of iterations, could be optimized to stop based on change(error), fine for now.", type=int, default=5) 
    parser.add_argument("--sm", help="smoothing parameter (if memory serves), potential changes/updates TBD", type=int, default=7)
    parser.add_argument("--m", help="frames to compare. larger is better, but math is worse. Simplest to evaluate against synthetic data", type=int, default=11)
    parser.add_argument("--mx", help="Part of odp logic for optimal pathing", type=int, default=9)
    parser.add_argument("--my", help="not used?", type=int, default=9)

    args = parser.parse_args()

    cores = args.cores
    nsteps = args.nsteps
    sm_param = args.sm
    m_frame = args.m
    mx = args.mx
    my = args.my

    fn = Path(args.fn)
    if not fn.exists():
        print(f"File not found {args.fn} {fn.absolute()}", file=sys.stderr)
        sys.exit(1)

    print(f"----starting {fn} ----")
    b = cs.from_h5file(fn)
    # for debug
    #    b.image_data = np.copy(b.image_data[0:60,:,:])

    nframes = b.image_data.shape[0]
    ndim = int(np.fix(nframes / cores))
    image_data2 = np.zeros((cores, ndim, b.image_data.shape[1], b.image_data.shape[2]))
    time_v = np.zeros((cores, ndim - m_frame + 1))
    s_index_list = []
    e_index_list = []
    e_index = 0
    for i in range(cores): # key dispatching logic where timeslices are dispersed across the compute
        # image_data2[i, :, :, :] = np.copy(b.image_data[ndim * i:ndim * (i + 1), :, :])
        s_index = np.max([0, e_index - m_frame + 1]) #actual timeslicing logic, start index, e is ending index
        e_index = s_index + ndim
        image_data2[i, :, :, :] = np.copy(b.image_data[s_index:e_index, :, :]) #image data for the respective timeslice
        time_v[i, :] = np.copy(b.time[s_index : e_index - m_frame + 1]) #associated time vector for sliced images
        s_index_list.append(s_index) #top level index tracking
        e_index_list.append(e_index)

    (Nt, Ny, Nx) = image_data2[0, :, :, :].shape  # Adjusted shape extraction, pulls dimensions of data prior to analysis
    #    nsteps = int(2 * np.log2(max(Nx, Ny) / 10) + 1)
    # Prepare arguments for parallel processing
    process_args = [
        (image_data2[i, :, :, :], nsteps, sm_param, m_frame, mx, my)
        for i in range(cores)
    ]

    # Use multiprocessing to process the imagesets in parallel
    with Pool(cores) as pool:  # Adjust the number of processes as needed
        results = pool.map(process_imageset, process_args)

    vx_all = []
    vy_all = []
    # Results for each imageset
    for i, result in enumerate(results):
        # vx, vy, vx_work, vy_work, image_warp, frame, steps, temp_y, iy, y_indices = result
        vx, vy = result
        vx_all.append(vx)
        vy_all.append(vy)
        print(f"Processed imageset {i}: vx shape = {vx.shape}, vy shape = {vy.shape}")
    # Combine results from all processes along the frame axis
    vx_stacked = np.concatenate(
        vx_all, axis=0
    )  # Combine all frames into a single array
    vy_stacked = np.concatenate(vy_all, axis=0)

    print("------- SAVING FILES ------")
    b.vx = vx_stacked
    b.vy = vy_stacked
    b.time_v = time_v

    # save_h5.from_object(b, path="./" + "vpara." + fn)
    scratch = Path(os.getenv("SCRATCH"))
    output = scratch / "hackathon" / "outputs" / f"vpara.{fn.name}"

    save_h5.from_object(b, path=output)
