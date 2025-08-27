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
    print('Processing: ', imageset.shape, nsteps, sm_param, m_frame, mx, my)
    return odp_idl.ODP(imageset, nsteps=nsteps, sm_param=sm_param, m_frame=m_frame, mx=mx, my=my)


def default_output_path(output) -> Path:
    cscratch = os.getenv("CSCRATCH")
    pscratch = os.getenv("PSCRATCH")
    if output is not None:
        out = Path(output)
    elif cscratch is not None:
        user = os.getenv("USER")
        out = Path(cscratch) / user / "bes_velocimetry_outputs"
    elif pscratch is not None:
        out = Path(pscratch) / "bes_velocimetry_outputs"
    else:
        raise FileNotFoundError("No output path given and no default for this system")

    out.mkdir(parents=True, exist_ok=True)
    return out


def reduce_spatial_res(vx, vy, R, Z, res_out):
    '''
    vx and vy should have shape (ntime, ny, nx)
    R, Z have shapes nx, ny
    res_out is a list with new [nx, ny]
    '''
    nx_out, ny_out = res_out  # resolution of the output velocity array
    px = nx // nx_out
    py = ny // ny_out
    # downsample R and Z
    R_down = np.zeros((nx_out))
    Z_down = np.zeros((ny_out))
    for i in range(nx_out):
        R_down[i] = R[i * px : (i + 1) * px].mean()
    for j in range(ny_out):
        Z_down[j] = Z[j * py : (j + 1) * py].mean()
    # downsample velocity
    nt = vx_stacked.shape[0]
    vx_down = np.zeros((nt, ny_out, nx_out))
    vy_down = np.zeros((nt, ny_out, nx_out))
    for i in range(ny_out):
        for j in range(nx_out):
            vx_sub = vx_stacked[:, i * py : (i + 1) * py, j * px : (j + 1) * px]
            vy_sub = vy_stacked[:, i * py : (i + 1) * py, j * px : (j + 1) * px]
            vx_down[:, i, j] = vx_sub.mean(axis=(1, 2))
            vy_down[:, i, j] = vy_sub.mean(axis=(1, 2))
    return vx_down, vy_down, R_down, Z_down


if __name__ == '__main__': # main():
    parser = argparse.ArgumentParser(description="check bes")
    parser.add_argument("--fn", help="hdf5 file name as the input", type=str, required=True)
    parser.add_argument("--out", help="Path for output file", type=str, default=None)
    parser.add_argument(
        "--cores", help="number of cores, runs 20x faster on NERSC when set to 256", type=int, default=10
    )
    parser.add_argument(
        "--nsteps",
        help="number of iterations, could be optimized to stop based on change(error), fine for now.",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--sm", help="smoothing parameter (if memory serves), potential changes/updates TBD", type=int, default=7
    )
    parser.add_argument(
        "--m",
        help="frames to compare. larger is better, but math is worse. Simplest to evaluate against synthetic data",
        type=int,
        default=11,
    )
    parser.add_argument("--mx", help="Part of odp logic for optimal pathing", type=int, default=9)
    parser.add_argument("--my", help="not used?", type=int, default=9)
    parser.add_argument("--res_out", help="output spatial resolution", type=int, nargs=2, default=[8, 8])

    args = parser.parse_args()

    cores = args.cores
    nsteps = args.nsteps
    sm_param = args.sm
    m_frame = args.m
    mx = args.mx
    my = args.my
    out = default_output_path(args.out)
    res_out = args.res_out

    fn = Path(args.fn)
    if not fn.exists():
        print(f"File not found {args.fn} {fn.absolute()}", file=sys.stderr)
        sys.exit(1)

    print(f"----starting {fn} ----")
    b = cs.from_h5file(fn)
    # for debug
    #    b.image_data = np.copy(b.image_data[0:60,:,:])

    (nframes, ny, nx) = b.images.shape  # images have shape (n_time, nZ, nR)
    ndim = int(np.fix(nframes / cores))
    image_data2 = np.zeros((cores, ndim, ny, nx))
    time_v = np.zeros((cores, ndim - m_frame + 1))
    s_index_list = []
    e_index_list = []
    e_index = 0
    for i in range(cores):  # key dispatching logic where timeslices are dispersed across the compute
        # image_data2[i, :, :, :] = np.copy(b.image_data[ndim * i:ndim * (i + 1), :, :])
        s_index = np.max([0, e_index - m_frame + 1])  # actual timeslicing logic, start index, e is ending index
        e_index = s_index + ndim
        # image data for the respective timeslice
        image_data2[i, :, :, :] = np.copy(b.images[s_index:e_index, :, :])
        time_v[i, :] = np.copy(b.time[s_index : e_index - m_frame + 1])  # associated time vector for sliced images
        s_index_list.append(s_index)  # top level index tracking
        e_index_list.append(e_index)

    # Adjusted shape extraction, pulls dimensions of data prior to analysis
    #    nsteps = int(2 * np.log2(max(nx, ny) / 10) + 1)
    # Prepare arguments for parallel processing
    process_args = [(image_data2[i, :, :, :], nsteps, sm_param, m_frame, mx, my) for i in range(cores)]

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
    vx_stacked = np.concatenate(vx_all, axis=0)  # Combine all frames into a single array
    vy_stacked = np.concatenate(vy_all, axis=0)

    # Convert velocity units from px/frame to m/s
    dR = (b.R[1] - b.R[0]) / 100  # cm -> m
    dZ = (b.Z[1] - b.Z[0]) / 100  # cm -> m
    dt = (b.time[1] - b.time[0]) / 1000  # ms -> s
    vx_stacked *= dR / dt
    vy_stacked *= dZ / dt

    # Downsample back to 8x8 resolution
    vx_down, vy_down, R_down, Z_down = reduce_spatial_res(vx_stacked, vy_stacked, b.R, b.Z, res_out)

    print("------- SAVING FILES ------")
    b.vx = vx_stacked
    b.vy = vy_stacked
    b.vx_down = vx_down
    b.vy_down = vy_down
    b.R_down = R_down
    b.Z_down = Z_down
    b.time_v = time_v

    output = out / f"vpara.{fn.name}"
    save_h5.from_object(b, path=output)
