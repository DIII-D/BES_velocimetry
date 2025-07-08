import bes_velocimetry.create_structure as cs
import bes_velocimetry.save_h5 as save_h5
import numpy as np
import bes_velocimetry.odp_idl as odp_idl
import argparse
from multiprocessing import Pool


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
    parser.add_argument("--fn", help="hdf5 file name as the input", type=str)
    args = parser.parse_args()
    fn = args.fn

    print(f"----starting {fn} ----")
    b = cs.from_h5file(fn)
    # for debug
    #    b.image_data = np.copy(b.image_data[0:60,:,:])

    cores = 10
    nsteps = 5
    sm_param = 7
    m_frame = 11
    mx = 9
    my = 9
    nframes = b.image_data.shape[0]
    ndim = int(np.fix(nframes / cores))
    image_data2 = np.zeros((cores, ndim, b.image_data.shape[1], b.image_data.shape[2]))
    time_v = np.zeros((cores, ndim - m_frame + 1))
    s_index_list = []
    e_index_list = []
    e_index = 0
    for i in range(cores):
        # image_data2[i, :, :, :] = np.copy(b.image_data[ndim * i:ndim * (i + 1), :, :])
        s_index = np.max([0, e_index - m_frame + 1])
        e_index = s_index + ndim
        image_data2[i, :, :, :] = np.copy(b.image_data[s_index:e_index, :, :])
        time_v[i, :] = np.copy(b.time[s_index : e_index - m_frame + 1])
        s_index_list.append(s_index)
        e_index_list.append(e_index)

    (Nt, Ny, Nx) = image_data2[0, :, :, :].shape  # Adjusted shape extraction
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

    save_h5.from_object(b, path="./" + "vpara." + fn)
