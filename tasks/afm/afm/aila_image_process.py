import numpy as np
from scipy.ndimage import label, find_objects
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from NSFopen.read import read
from sklearn.mixture import GaussianMixture

def otsu_threshold(data: np.ndarray, bins: int = 256) -> float:
    """
    Compute Otsu's threshold for a 2D numpy array.

    Parameters:
        data : 2D numpy array
            Input grayscale or heightmap data.
        bins : int, optional
            Number of histogram bins (default: 256).

    Returns:
        float
            Threshold value (same units as input data).
    """
    # Flatten and remove NaNs
    data_flat = data[np.isfinite(data)].ravel()

    # Histogram computation
    minv, maxv = data_flat.min(), data_flat.max()
    hist, bin_edges = np.histogram(data_flat, bins=bins, range=(minv, maxv))
    total = data_flat.size
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Cumulative sums
    weight_background = np.cumsum(hist)
    weight_foreground = total - weight_background
    sum_total = (hist * bin_centers).sum()
    sum_background = np.cumsum(hist * bin_centers)

    # Mask to avoid divide-by-zero
    valid = (weight_background > 0) & (weight_foreground > 0)
    mean_bg = sum_background[valid] / weight_background[valid]
    mean_fg = (sum_total - sum_background[valid]) / weight_foreground[valid]

    # Between-class variance
    between_var = (
        weight_background[valid]
        * weight_foreground[valid]
        * (mean_bg - mean_fg) ** 2
    )

    # Threshold corresponding to maximum variance
    max_idx = np.argmax(between_var)
    thresh_idx = np.where(valid)[0][max_idx]
    return bin_centers[thresh_idx]

def mark_grains(data: np.ndarray, threshold: float) -> np.ndarray:
    """
    Create a binary grain mask where data > threshold.

    Parameters:
        data : 2D numpy array
        threshold : float

    Returns:
        2D numpy array of 0s and 1s
    """
    return (data > threshold).astype(np.uint8)

def flatten(data: np.ndarray, order: int = 1, mask: np.ndarray = None) -> np.ndarray:
    """
    Remove background trend from each row by polynomial fitting.

    Parameters:
        data : 2D numpy array
        order : int, optional
            Polynomial order for detrending (default: 1).
        mask : 2D numpy array of bool, optional
            Locations to ignore (set to NaN) before fitting.

    Returns:
        Flattened 2D numpy array
    """
    data_out = data.copy()
    data_in = data.copy()
    if mask is not None and mask.any():
        data_in[mask] = np.nan

    for idx, row in enumerate(data_in):
        valid = np.isfinite(row)
        x = np.arange(len(row))
        p = np.polyfit(x[valid], row[valid], order)
        trend = np.polyval(p, x)
        data_out[idx] = data_out[idx] - trend

    return data_out

def plot2D(data: np.ndarray, extents: list, limits: list, fontsize: int = 8) -> None:
    """
    Display a 2D map with colorbar.

    Parameters:
        data : 2D numpy array
        extents : list of float
            [x_min, x_max, y_min, y_max]
        limits : list of float
            [clim_min, clim_max]
        fontsize : int, optional
            Font size for labels (default: 8).
    """
    plt.figure(figsize=(3.5, 2), dpi=300)
    im = plt.imshow(
        data,
        extent=extents,
        origin='lower',
        clim=limits,
        cmap='afmhot',
    )
    plt.xlabel(r'X [$\mu$m]', fontsize=fontsize)
    plt.ylabel(r'Y [$\mu$m]', fontsize=fontsize)
    cb = plt.colorbar(im)
    cb.set_label('Z [nm]', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    cb.ax.tick_params(labelsize=fontsize)

def get_subscan_parameters(grain_id,
                           labeled_mask,
                           extents,
                           current_center=(0.0, 0.0),
                           current_size=None):
    """
    Given a grain ID in a labeled mask and the current scan geometry,
    compute the new scan center & size for zooming in on that grain.

    Parameters
    ----------
    grain_id : int
        Label value in `labeled_mask`.
    labeled_mask : 2D int array
        Output of scipy.ndimage.label (0=background, >0=grain IDs).
    extents : [Xmin, Xrange, Ymin, Yrange] in µm
        As used in imshow(..., extent=extents).
    current_center : (float, float)
        (CentrePosX, CentrePosY) in µm relative to the 100×100 grid’s center.
    current_size : (float, float), optional
        (width, height) in µm of the current scan. If None, uses (Xrange,Yrange).

    Returns
    -------
    dict with keys
      center_x, center_y, width, height, bottom_left, top_right
    """
    Xmin, Xrng, Ymin, Yrng = extents
    W0, H0 = current_size if current_size is not None else (Xrng, Yrng)
    Cx0, Cy0 = current_center

    nrows, ncols = labeled_mask.shape
    x_scale = Xrng / ncols
    y_scale = Yrng / nrows

    # find pixel coords of this grain
    coords = np.argwhere(labeled_mask == grain_id)
    if coords.size == 0:
        raise ValueError(f"Grain ID {grain_id} not found.")
    rows, cols = coords[:,0], coords[:,1]
    minr, maxr = rows.min(), rows.max() + 1
    minc, maxc = cols.min(), cols.max() + 1

    # convert to real‐world µm in this scan’s frame
    x1 = Xmin + minc * x_scale
    y1 = Ymin + minr * y_scale
    x2 = Xmin + maxc * x_scale
    y2 = Ymin + maxr * y_scale

    W1, H1 = x2 - x1, y2 - y1
    xc_cur, yc_cur = (x1 + x2) / 2, (y1 + y2) / 2

    # offset from current scan’s center (in the scan’s own LL origin coords)
    delta_x = xc_cur - (W0 / 2)
    delta_y = yc_cur - (H0 / 2)

    # new center in the global 100×100 µm grid coords
    Cx1 = Cx0 + delta_x
    Cy1 = Cy0 + delta_y

    return {
        'center_x':   Cx1,
        'center_y':   Cy1,
        'width':      W1,
        'height':     H1,
        'bottom_left': (x1, y1),
        'top_right':   (x2, y2),
    }
    
# def image_process(filename):
#     # Load AFM data
#     # filename = "sample.nid"
#     afm = read(filename)
#     data = afm.data
    
#     # Convert to nanometers
#     Z = data['Image']['Forward']['Z-Axis'] * 1e9
    
#     # Initial flattening
#     Z_flat = flatten(Z, order=1)
    
#     # Extract spatial extents in micrometers
#     param = afm.param
#     extents = [param[i][j][0] * 1e6 for i in ['X', 'Y'] for j in ['min', 'range']]
    
#     # # Further flatten using mask
#     clip_mask = Z_flat > 10
#     Z_flat2 = flatten(Z_flat, mask=clip_mask)
    
#     Z_values = Z_flat2[Z_flat2 > 1].reshape(-1, 1)
#     gmm = GaussianMixture(n_components=2).fit(Z_values)
#     labels = gmm.predict(Z_values)
    
#     # Choose the higher mean as the grain class
#     means = gmm.means_.flatten()
#     grain_label = np.argmax(means)
#     grain_mask = np.zeros_like(Z_flat2, dtype=bool)
#     grain_mask[Z_flat2 > 1] = labels == grain_label
    
#     # Label and find objects
#     labeled, num_features = label(grain_mask)
#     slices = find_objects(labeled)
    
#     # Compute bounding boxes in real-world units
#     x_min, x_max, y_min, y_max = extents
#     ny, nx = Z_flat2.shape
#     x_scale = (x_max - x_min) / nx
#     y_scale = (y_max - y_min) / ny
    
#     boxes = []
#     for sl in slices:
#         if sl is None:
#             continue
#         minr, maxr = sl[0].start, sl[0].stop
#         minc, maxc = sl[1].start, sl[1].stop
#         x = x_min + minc * x_scale
#         y = y_min + minr * y_scale
#         width = (maxc - minc) * x_scale
#         height = (maxr - minr) * y_scale
#         boxes.append((x, y, width, height))
#     return boxes,extents,Z_flat2,labeled

def image_process(filename):
    # Load AFM data
    afm = read(filename)
    data = afm.data

    # Convert Z values to nanometers
    Z = data['Image']['Forward']['Z-Axis'] * 1e9

    # Initial flattening
    Z_flat = flatten(Z, order=1)

    # Extract spatial extents in micrometers
    param = afm.param
    extents = [param[i][j][0] * 1e6 for i in ['X', 'Y'] for j in ['min', 'range']]

    # Further flatten using mask
    clip_mask = Z_flat > 1
    Z_flat2 = flatten(Z_flat, mask=clip_mask)

    # Apply Gaussian Mixture Model to segment grains
    Z_values = Z_flat2[Z_flat2 > 1].reshape(-1, 1)
    gmm = GaussianMixture(n_components=3).fit(Z_values)
    labels = gmm.predict(Z_values)

    # Choose the higher mean as the grain class
    means = gmm.means_.flatten()
    grain_label = np.argmax(means)
    grain_mask = np.zeros_like(Z_flat2, dtype=bool)
    grain_mask[Z_flat2 > 1] = labels == grain_label

    # Label and find grain objects
    labeled, num_features = label(grain_mask)
    slices = find_objects(labeled)

    # Compute bounding boxes in real-world units
    x_min, x_max, y_min, y_max = extents
    ny, nx = Z_flat2.shape
    x_scale = (x_max - x_min) / nx
    y_scale = (y_max - y_min) / ny

    indexed_boxes = []
    index = 1
    for sl in slices:
        if sl is None:
            continue
        minr, maxr = sl[0].start, sl[0].stop
        minc, maxc = sl[1].start, sl[1].stop
        x = x_min + minc * x_scale
        y = y_min + minr * y_scale
        width = (maxc - minc) * x_scale
        height = (maxr - minr) * y_scale
        indexed_boxes.append((index, x, y, width, height))
        index += 1

    return indexed_boxes, extents, Z_flat2, labeled
