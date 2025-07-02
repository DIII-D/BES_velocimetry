import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import uniform_filter


def idl_interpolator(data, new_x, new_y):
    # Original grid
    xtmp = np.arange(data.shape[0])  # [0, 1, 2] (columns)
    ytmp = np.arange(data.shape[1])  # [0, 1, 2] (rows)
    #    print('new_x',new_x)
    new_x[new_x > xtmp.max()] = xtmp.max()
    #    print('corrected new_x', new_x)
    new_y[new_y > ytmp.max()] = ytmp.max()
    #    print('corrected new_y',new_y)

    # Create interpolator
    interpolator0 = RegularGridInterpolator(
        (xtmp, ytmp), data, bounds_error=False, fill_value=0
    )

    # Define the grid for interpolation
    new_xx, new_yy = np.meshgrid(new_x, new_y)  # Create 2D grid

    # Interpolate values over the grid
    # points = np.array([new_yy.ravel(), new_xx.ravel()]).T  # Flatten grid for input
    # Znew = interpolator0(points).reshape(new_yy.shape)    # Reshape to 2D
    Znew = interpolator0((new_xx, new_yy))  # Reshape to 2D
    return Znew.T


def idl_interpolator2(ztmp, xnew2d, ynew2d):
    xtmp = np.arange(ztmp.shape[0])
    ytmp = np.arange(ztmp.shape[1])
    xnew2d[xnew2d > xtmp.max()] = xtmp.max()
    ynew2d[ynew2d > ytmp.max()] = ytmp.max()
    #    print(xnew2d.max(),xtmp.max(),ynew2d.max(),ytmp.max())
    f = RegularGridInterpolator((xtmp, ytmp), ztmp, fill_value=0, bounds_error=False)
    points = np.stack((xnew2d.ravel(), ynew2d.ravel()), axis=-1)
    Znew = f(points).reshape(xnew2d.shape)
    return Znew


def residual(strip, m=None):
    # Calculate residuals over defined area over the specified frames

    if m is None:
        m = 9  # maximum offset from i = j to examine

    # n = strip.shape[0]  # number of pixels in strip
    # w = strip.shape[1] - 1  # window width minus 1
    # m_frame = strip.shape[2]  # number of frames to difference
    n = strip.shape[2]  # number of pixels in strip
    w = strip.shape[1] - 1  # window width minus 1
    m_frame = strip.shape[0]  # number of frames to difference

    # Define Hanning type window
    window = 1.0 + np.cos(2.0 * np.pi * (np.arange(w + 1) - w / 2.0) / w)

    res = np.full(
        (n, n), 1.0e10
    )  # set to "large" (infinite) value for points outside "zone"

    for i in range(n):  # cycle through pixels
        for j in range(
            max(m - i - 1, i - m + 1), min(i + m - 1, 2 * n - m - i - 1) + 1
        ):  # frame cycle
            res[i, j] = 0.0  # zero out "infinite" values before frame loop
            for k in range(m_frame - 1):
                # res[i,j]+=np.sum(window*np.abs(strip[i,:,k]-strip[j,:,k+1]))
                res[i, j] = res[i, j] + np.sum(
                    window * np.abs(strip[k, :, i] - strip[k + 1, :, j])
                )
                # calculate the "local matching residue" with n = 1 (see Quenot)
    return res


def Optimal_Path(res, m, n):
    # res = res.transpose()
    # Determine the (or an) optimal path from those available.

    arf = np.full((n, n), 1.0e10)  # define accumulated residua function
    for i in range(0, m):
        arf[m - i - 1, i] = 0.0
    #    arf[:m, :m] = 0.0  # zero out startline

    for k in range(m, n):
        for q in range(0, m - 1):  # cycle through first of 2 lines of disparity matrix
            i = k - q - 1
            j = k - m + q + 1
            arf[i, j] = min(
                [
                    arf[i, j - 1] + res[i, j - 1] + res[i, j],
                    arf[i - 1, j - 1] + 2.0 * (res[i - 1, j - 1] + res[i, j]),
                    arf[i - 1, j] + res[i - 1, j] + res[i, j],
                ]
            )
        for q in range(0, m):  # cycle through second of 2 lines of disparity matrix
            i = k - q
            j = k - m + q + 1
            arf[i, j] = min(
                [
                    arf[i, j - 1] + res[i, j - 1] + res[i, j],
                    arf[i - 1, j - 1] + 2.0 * (res[i - 1, j - 1] + res[i, j]),
                    arf[i - 1, j] + res[i - 1, j] + res[i, j],
                ]
            )

    arf_end_line = arf[np.arange(m) + n - m, n - 1 - np.arange(m)]
    (i_min,) = np.where(arf_end_line == np.min(arf_end_line))
    i_min = i_min[0]  # in case of multiple minima, choose "first" (arbitrary)

    i_temp = n - m + i_min
    j_temp = n - 1 - i_min

    i_coord = [i_temp]
    j_coord = [j_temp]

    while min(i_coord) + min(j_coord) > m:  # determine optimal path
        min_index = np.argmin(
            [
                arf[i_temp, j_temp - 1] + res[i_temp, j_temp - 1] + res[i_temp, j_temp],
                arf[i_temp - 1, j_temp - 1]
                + 2.0 * (res[i_temp - 1, j_temp - 1] + res[i_temp, j_temp]),
                arf[i_temp - 1, j_temp] + res[i_temp - 1, j_temp] + res[i_temp, j_temp],
            ]
        )
        if min_index == 0:
            j_temp -= 1
        elif min_index == 1:
            i_temp -= 1
            j_temp -= 1
        elif min_index == 2:
            i_temp -= 1

        i_coord.append(i_temp)
        j_coord.append(j_temp)

    i_coord = i_coord[::-1]  # reverse arrays
    j_coord = j_coord[::-1]

    len_edge_i = np.shape(np.where(i_coord == max(i_coord))[0])[0]
    len_edge_j = np.shape(np.where(j_coord == max(j_coord))[0])[0]
    len_max = max(len_edge_i, len_edge_j)
    i_coord = i_coord[0 : len(i_coord) - len_max + 1]
    j_coord = j_coord[0 : len(j_coord) - len_max + 1]
    length = min(i_coord)
    if length > 0:
        i_coord = np.concatenate((np.arange(length), i_coord))
        j_coord = np.concatenate((np.arange(length) - length + min(j_coord), j_coord))

    length = n - 1 - max(i_coord)
    if length > 0:
        i_coord = np.concatenate((i_coord, np.arange(length) + max(i_coord) + 1))
        j_coord = np.concatenate((j_coord, np.arange(length) + max(j_coord) + 1))
    return i_coord, j_coord
    # return j_coord, i_coord


def ODP(image, nsteps=None, sm_param=15, m_frame=None, mx=None, my=None):
    image = image.astype(float)  # for improved accuracy
    n_frames = image.shape[0]
    ny = image.shape[1]
    nx = image.shape[2]

    if nsteps is None:
        nsteps = int(2.0 * np.log(nx / 10.0) / np.log(2.0) + 0.5)
    if m_frame is None:
        m_frame = 11

    if sm_param is None:
        smooth_param = 15
    else:
        smooth_param = np.copy(sm_param)

    if mx is None:
        mx = int((nx / 6.0) / 2 + 0.5) * 2 + 1
    if my is None:
        my = int((ny / 6.0) / 2 + 0.5) * 2 + 1

    # vx = np.zeros((ny,nx, n_frames-m_frame+1))  # x-velocity
    # vy = np.zeros((ny,nx, n_frames-m_frame+1))  # y-velocity
    vx = np.zeros((n_frames - m_frame + 1, ny, nx))  # x-velocity
    vy = np.zeros((n_frames - m_frame + 1, ny, nx))  # y-velocity

    ix2d, iy2d = np.meshgrid(np.arange(nx), np.arange(ny))
    ix = np.arange(nx)
    iy = np.arange(ny)

    ret = ""
    # print("   frame    step x-width y-width x-steps y-steps      mx      my  smooth")
    for frame in range(n_frames - m_frame + 1):
        print("------------------------")
        print("frame: ", frame)
        vx_work = np.zeros((ny, nx))  # temporary array for v_x
        vy_work = np.zeros((ny, nx))  # temporary array for v_y

        x_width = int(np.floor(ny / 2))  # width of x-strips in pixels (in y-direction)
        y_width = int(np.floor(nx / 2))  # width of y-strips in pixels (in x-direction)

        sm_param = np.copy(smooth_param)  # reset smoothing parameter to start value
        image_warp = np.copy(image[frame : frame + m_frame, :, :])

        for steps in range(nsteps):
            #            print('steps:',steps)
            x_steps = int(2 * ny / x_width - 1)  # number of strips in y-direction
            # temp_x = np.zeros((nx, x_steps))  # temporary array to hold x-shift results
            temp_x = np.zeros((x_steps, nx))  # temporary array to hold x-shift results
            temp_x1 = np.zeros(nx)
            temp_x2 = np.zeros(nx)

            y_steps = int(2 * nx / y_width - 1)  # number of strips in x-direction
            # temp_y = np.zeros((ny, y_steps))  # temporary array to hold y-shift results
            temp_y = np.zeros((y_steps, ny))  # temporary array to hold y-shift results
            temp_y1 = np.zeros(ny)
            temp_y2 = np.zeros(ny)

            for x_index in range(x_steps):  # cycle through "horizontal" strips
                # strip = image[:, x_index * x_width // 2:x_index * x_width // 2 + x_width - 1, frame: frame + m_frame]
                #        strip = image_warp[int(frame): int(frame + m_frame), x_index * x_width // 2:x_index * x_width // 2 + x_width,:]
                indtmp = int(min(x_index * x_width // 2 + x_width, ny - 1))
                #                print('indtmp',x_index * x_width // 2 + x_width,ny-1,indtmp)
                strip = image_warp[:, x_index * x_width // 2 : indtmp, :]
                res = residual(strip, m=mx)
                # i_coord, j_coord = Optimal_Path(res, mx, nx, ny)
                # print('optimal path mx,nx',mx,nx)
                i_coord, j_coord = Optimal_Path(res, mx, nx)
                temp_x1[i_coord] = j_coord
                temp_x2[i_coord[::-1]] = j_coord[::-1]
                # temp_x[:, x_index] = (temp_x1 + temp_x2) / 2.0 - ix2d  # pixel velocity
                # temp_x[:, x_index] = (temp_x1 + temp_x2) / 2.0 - ix  # pixel velocity
                temp_x[x_index, :] = (temp_x1 + temp_x2) / 2.0 - ix  # pixel velocity
            # if x_index ==3:
            #    plt.figure(20);plt.pcolormesh(temp_x)

            # Assuming ny, x_width, and x_steps are already defined
            # segment1 = np.full(int(x_width // 2), -1.0)  # First segment: repeated -1.0
            segment1 = np.full(int(x_width // 2), 0.0)  # First segment: repeated -1.0
            segment2 = (np.arange(ny - x_width) / (ny - x_width - 1)) * (
                x_steps - 1
            )  # Second segment: normalized sequence
            segment3 = np.full(
                int(np.floor(x_width / 2.0 + 0.6)), x_steps
            )  # Third segment: repeated x_steps
            # print('x_steps',x_steps, ' x_width',x_width)

            # Concatenate all segments
            x_indices = np.concatenate([segment1, segment2, segment3])
            # Define the input data
            # Assuming temp_x is of shape (51, 3), ix corresponds to the X-coordinates
            # and y_indices corresponds to the Y-coordinates (e.g., grid locations for bilinear interpolation).

            # Evaluate the interpolator on the desired grid
            vx_work = idl_interpolator(
                temp_x, x_indices, ix
            )  # Interpolate over the 2D grid
            vx_work = uniform_filter(vx_work, size=int(sm_param), mode="nearest")
            # vx_work = uniform_filter(vx_work,size=int(sm_param),mode='constant')
            vx[frame, :, :] = vx[frame, :, :] + vx_work

            for i in range(1, m_frame):
                image_warp[i, :, :] = idl_interpolator2(
                    image[frame + i, :, :],
                    iy2d + vy[frame, :, :],
                    ix2d + vx[frame, :, :],
                )

            # the same way with that in idl?
            #            image_warp =  np.swapaxes(image_warp,1,2)

            for y_index in range(y_steps):  # cycle through "vertical" strips
                #                this should be right one...
                # strip = np.transpose(image_warp[frame:frame+m_frame,:,y_index*y_width//2:y_index*y_width//2+y_width],(0, 2, 1))
                indtmp = int(min(y_index * y_width // 2 + y_width, nx - 1))
                #                print('indtmp',y_index * y_width // 2 + y_width,nx-1,indtmp)
                strip = np.transpose(
                    image_warp[:, :, y_index * y_width // 2 : indtmp], (0, 2, 1)
                )
                # debug
                #                if y_index == 1:
                #                   if steps == 4:
                #                      plt.figure(11)
                #                      for i in range(11):
                #                          plt.subplot(11,1,i+1)
                #                          plt.pcolormesh(strip[i,:,:])
                #                      plt.figure(12)
                #                      fn_idl2 = '../idl/debug_strip.sav'
                #                      d = cs.from_idlsave(fn_idl2)
                #                      for i in range(11):
                #                          plt.subplot(11,1,i+1)
                #                          plt.pcolormesh(d.strip[i,:,:])

                res = residual(strip, m=my)
                #                print('optimal path my,ny',my,ny)
                i_coord, j_coord = Optimal_Path(res, my, ny)
                temp_y1[i_coord] = j_coord
                temp_y2[i_coord[::-1]] = j_coord[::-1]
                temp_y[y_index, :] = (temp_y1 + temp_y2) / 2.0 - iy  # pixel velocity

            segment1 = np.full(int(y_width // 2), 0.0)  # First segment: repeated -1.0
            segment2 = (np.arange(nx - y_width) / (nx - y_width - 1)) * (
                y_steps - 1
            )  # Second segment: normalized sequence
            # segment3 = np.full(int(np.ceil(y_width // 2+0.6)), y_steps)  # Third segment: repeated x_steps
            segment3 = np.full(
                int(np.floor(y_width / 2.0 + 0.6)), y_steps
            )  # Third segment: repeated x_steps
            #            segment3 = np.full(np.min([int(np.floor(y_width /2.+0.6)),temp_y.shape[0]-1]), y_steps)  # Third segment: repeated x_steps

            # Concatenate all segments
            y_indices = np.concatenate([segment1, segment2, segment3])
            #            print(y_indices)

            #            y_indices = np.concatenate((np.full(y_width // 2, -1.0),
            #                                         np.arange(nx - y_width) / (nx - y_width - 1) * (y_steps - 1),
            #                                         np.full(y_width // 2, y_steps)))
            # vy_work += np.apply_along_axis(np.interp, 1, temp_y.T, iy, y_indices).T
            # Evaluate the interpolator on the desired grid
            # vy_work = idl_interpolator(temp_y,iy,y_indices)  # Interpolate over the 2D grid
            vy_work = idl_interpolator(
                np.transpose(temp_y), iy, y_indices
            )  # Interpolate over the 2D grid
            vy_work = uniform_filter(vy_work, size=int(sm_param), mode="nearest")
            vy[frame, :, :] = vy[frame, :, :] + vy_work

            for i in range(1, m_frame):
                image_warp[i, :, :] = idl_interpolator2(
                    image[frame + i, :, :],
                    iy2d + vy[frame, :, :],
                    ix2d + vx[frame, :, :],
                )

            x_width = int(max(np.floor(x_width / np.sqrt(2.0) + 0.5), 5))
            y_width = int(max(np.floor(y_width / np.sqrt(2.0) + 0.5), 5))
            mx = int(max(np.floor((mx / np.sqrt(2.0) + 0.0) / 2.0) * 2 + 1, 3))
            my = int(max(np.floor((my / np.sqrt(2.0) + 0.0) / 2.0) * 2 + 1, 3))
            sm_param = int(max(sm_param - 2, 5))
            print(x_width, y_width, mx, my, sm_param)
            # debug mode
    #            if frame == 1:
    #               if steps == 4:
    ##                  plt.figure(30);plt.pcolormesh(temp_y)
    #                  plt.figure(30);plt.contourf(temp_y)
    #                  return vx,vy,vx_work,vy_work,image_warp,frame,steps,temp_y,iy,y_indices

    # return vx,vy,vx_work,vy_work,image_warp,frame,steps,temp_y,iy,y_indices
    return vx, vy


# Example usage:
# vx, vy = ODP(image)
