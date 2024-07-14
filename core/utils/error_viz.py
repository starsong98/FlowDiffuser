"""
Solely for visualizing error maps.

Visualize EPE error maps.

Similarly to Spring & KITTI benchmark sites, blue will be low error, red will be high error.
With sparse GT dsets, following KITTI benchmark site, areas w/o GT measured will be masked out in black.

Later work will involve porting the KITTI original devkit code, or somehow calling matlab thru python.
"""
# import dependencies
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def visualize_error_map(epe, valid=None, epe_max=48, cmap='bwr'):
    """
    Parameters:
    * epe: numpy ndarray, dimensions [H, W], dtype float.
    * valid: numpy ndarray, dimensions [H, W], boolean. 0 is invalid and to be ignored; 1 is valid and to be evaluated.
    * cmap: colormap for error map visualization.
    """
    #cm_fn = plt.get_cmap('bwr')
    #cm_fn = plt.get_cmap('Reds')
    cm_fn = plt.get_cmap(cmap)
    epe_map = cm_fn(epe / epe_max)
    #print(epe_map.shape)
    if valid is not None:
        epe_map[~valid] = 0
    epe_map = np.uint8(epe_map[:,:,0:3] * 255)  # because the 4th channel is alpha (transparency)
    return epe_map
