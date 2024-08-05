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

from .flow_viz import flow_uv_to_colors

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


# something that visualizes output flow map with same normalization as gt flow map
# planning needed
def compare_flow_viz(
    out_flow_uv,
    gt_flow_uv,
    clip_flow=None,
    convert_to_bgr=False
):
    """
    Returns visualizations of some predicted flow alongside its gt flow.
    But normalize the output flow the same as gt flow. This wasn't really done before.

    Args:
        out_flow_uv (np.ndarray): Flow YV image of shape [H, W, 2]
        gt_flow_uv (np.ndarray): Flow YV image of shape [H, W, 2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        out_flow_viz (np.ndarray): Flow visualization image of shape [H,W,3]
        gt_flow_viz (np.ndarray): Flow visualization image of shape [H,W,3]
    """

    assert out_flow_uv.ndim == 3, 'input predicted flow must have three dimensions'
    assert out_flow_uv.shape[2] == 2, 'input predicted flow must have shape [H,W,2]'
    assert gt_flow_uv.ndim == 3, 'input GT flow must have three dimensions'
    assert gt_flow_uv.shape[2] == 2, 'input GT flow must have shape [H,W,2]'

    if clip_flow is not None:
        out_flow_uv = np.clip(out_flow_uv, 0, clip_flow)
        gt_flow_uv = np.clip(gt_flow_uv, 0, clip_flow)

    out_u = out_flow_uv[:,:,0]
    out_v = out_flow_uv[:,:,1]
    gt_u = gt_flow_uv[:,:,0]
    gt_v = gt_flow_uv[:,:,1]
    
    rad = np.sqrt(np.square(gt_u) + np.square(gt_v))
    rad_max = np.max(rad)
    epsilon = 1e-5

    out_u = out_u / (rad_max + epsilon)
    out_v = out_v / (rad_max + epsilon)
    gt_u = gt_u / (rad_max + epsilon)
    gt_v = gt_v / (rad_max + epsilon)

    out_flow_viz = flow_uv_to_colors(out_u, out_v, convert_to_bgr)
    gt_flow_viz = flow_uv_to_colors(gt_u, gt_v, convert_to_bgr)

    return out_flow_viz, gt_flow_viz