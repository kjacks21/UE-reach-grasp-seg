import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.decomposition import PCA
from scipy.ndimage import median_filter
from sklearn.preprocessing import MinMaxScaler
from scipy.io import loadmat
from numpy.random import uniform, normal
import os
from collections import defaultdict
import copy
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
import csv
from scipy.spatial.distance import euclidean

def compute_vel_signed(d_vals, vel):
    
    diffs = np.diff(d_vals)
    diffs[diffs<0] = -1
    diffs[diffs>0] = 1
    return diffs * vel

def viz_seg_result(f_path, x, y, z, vel, w_vals, se, viz_dist_thresh=True, w_thresh_indx=None):
    """
    Viz only one seg result
    """
    fig = plt.figure(figsize=(15,5))

    ax = fig.add_subplot(1,3,1, projection='3d')

    ax.scatter(x[se], y[se], z[se], marker='o', c='b', s=45)
    ax.scatter(x[-1], y[-1], z[-1], marker='x', c='r', s=55)
    ax.plot(x, y, z, color='k', marker='D', markerfacecolor='r', markevery=[0], linewidth=1)

    ax.set_title('(b) Position', y=1.08)

    ax.set_xticks(ax.get_xticks()[::2])
    ax.set_yticks(ax.get_yticks()[::2])
    ax.set_zticks(ax.get_zticks()[::2])

    ax = fig.add_subplot(1,3,2)

    ax.plot(w_vals, '-k')
    ax.axhline(w_vals[0], color='r', linestyle='-', label='Start')
    ax.axhline(w_vals[-1], color='r', linestyle=':', label='End')
    if viz_dist_thresh:
        ax.axvline(w_thresh_indx, color='g', linestyle='-.', label='$d_s$ Distance Threshold')
    ax.axvline(se, color='b', linestyle='--', label='Segmentation')
    ax.set_xlabel('Frames', fontsize=12)
    ax.set_ylabel('Displacement $d$ (unspecified unit)', fontsize=12)

    ax.set_title('(b) Segmentation Along $d$ (Eq. 3)')
    ax.legend()

    ax = fig.add_subplot(1,3,3)

    ax.plot(vel, '-k')

    ax.set_title('(b) Velocity Profile')
    ax.set_xlabel('Frames', fontsize=12)
    ax.set_ylabel('unspecified unit/s', fontsize=12)
    if viz_dist_thresh:
        ax.axvline(w_thresh_indx, color='g', linestyle='-.', label='$d_s$ Distance Threshold')
    ax.axvline(se, c='b', ls='--', label='Segmentation')
    ax.legend()
    ax.tick_params()

    plt.tight_layout()
    plt.savefig(f_path, bbox_inches='tight')
    plt.close()

    ####################

def viz_all_seg_result(f_path, x, y, z, vel, w_vals, se_list):
    """
    Viz all seg results
    """
    marker_lookup = {
        'jackson' : {'marker':'o', 'c':'b', 'linestyle':(0, (5, 10))}, # loosely dashed
        'sakai' : {'marker':'s', 'c':'g', 'linestyle':(0, (3, 5, 1, 5))}, # dashdotted
        '50_percent' : {'marker':'*', 'c':'c', 'linestyle':(0, (3, 5, 1, 5, 1, 5))}, # dashdotdotted
        'kinematic' : {'marker':'p', 'c':'m', 'linestyle':'dotted'}, # dotted
        'jackson_no_d' : {'marker':'P', 'c':'darkorange', 'linestyle':'dashed'} # dashed
    }

    fig = plt.figure(figsize=(15,6))

    ax = fig.add_subplot(1,3,1, projection='3d')

    for i in se_list: 
        ax.scatter(x[i[1]], y[i[1]], z[i[1]], marker=marker_lookup[i[0]]['marker'], c=marker_lookup[i[0]]['c'], s=45, label=i[0])
    ax.scatter(x[-1], y[-1], z[-1], marker='x', c='r', s=55)
    ax.plot(x, y, z, color='k', marker='D', markerfacecolor='r', markevery=[0], linewidth=1)

    ax.set_title('Position', y=1.08)
    ax.legend()

    ax.set_xticks(ax.get_xticks()[::2])
    ax.set_yticks(ax.get_yticks()[::2])
    ax.set_zticks(ax.get_zticks()[::2])

    ax = fig.add_subplot(1,3,2)

    ax.plot(w_vals, '-k')
    ax.axhline(w_vals[0], color='r', linestyle='-', label='Start')
    ax.axhline(w_vals[-1], color='r', linestyle=':', label='End')
    for i in se_list: 
        ax.axvline(i[1], color=marker_lookup[i[0]]['c'], linestyle=marker_lookup[i[0]]['linestyle'], label=i[0], alpha=0.75)
    # ax.axvline(se, color='b', linestyle='--', label='Segmentation')
    ax.set_xlabel('Frames', fontsize=12)
    ax.set_ylabel('Displacement $d$ (unspecified unit)', fontsize=12)

    ax.set_title('Trajectory Displacement')
    ax.legend()

    ax = fig.add_subplot(1,3,3)

    ax.plot(vel, '-k')

    ax.set_title('Velocity Profile')
    ax.set_xlabel('Frames', fontsize=12)
    ax.set_ylabel('unspecified unit/s', fontsize=12)
    
    for i in se_list: 
        ax.axvline(i[1], color=marker_lookup[i[0]]['c'], linestyle=marker_lookup[i[0]]['linestyle'], label=i[0], alpha=0.75)
    # ax.axvline(se, c='b', ls='--', label='Segmentation')
    ax.legend()
    ax.tick_params()

    plt.tight_layout()
    plt.savefig(f_path, bbox_inches='tight')
    plt.close()

    ####################

def fourplot_viz_seg_result(f_path, x, y, z, vel, w_vals, traj_displacement, se, viz_dist_thresh=True, w_thresh_indx=None):
    """
    Viz only one seg result
    """
    fig = plt.figure(figsize=(15,5))

    ax = fig.add_subplot(2,3,1, projection='3d')

    ax.scatter(x[se], y[se], z[se], marker='o', c='b', s=45)
    ax.scatter(x[-1], y[-1], z[-1], marker='x', c='r', s=55)
    ax.plot(x, y, z, color='k', marker='D', markerfacecolor='r', markevery=[0], linewidth=1)

    ax.set_title('(b) Position', y=1.08)

    ax.set_xticks(ax.get_xticks()[::2])
    ax.set_yticks(ax.get_yticks()[::2])
    ax.set_zticks(ax.get_zticks()[::2])

    ax = fig.add_subplot(2,3,2)

    ax.plot(traj_displacement, '-k')
    ax.axhline(traj_displacement[0], color='r', linestyle='-', label='Start')
    ax.axhline(traj_displacement[-1], color='r', linestyle=':', label='End')
    if viz_dist_thresh:
        ax.axvline(w_thresh_indx, color='g', linestyle='-.', label='$d_s$ Distance Threshold')
    ax.axvline(se, color='b', linestyle='--', label='Segmentation')
    ax.set_xlabel('Frames', fontsize=12)
    ax.set_ylabel('Displacement $d$ (unspecified unit)', fontsize=12)

    ax.set_title('(b) Segmentation Along $d$ (Eq. 3)')
    ax.legend()

    ax = fig.add_subplot(2,3,3)

    ax.plot(vel, '-k')

    ax.set_title('(b) Velocity Profile')
    ax.set_xlabel('Frames', fontsize=12)
    ax.set_ylabel('unspecified unit/s', fontsize=12)
    if viz_dist_thresh:
        ax.axvline(w_thresh_indx, color='g', linestyle='-.', label='$d_s$ Distance Threshold')
    ax.axvline(se, c='b', ls='--', label='Segmentation')
    ax.legend()
    ax.tick_params()

    ax = fig.add_subplot(2,3,4)

    signed_vel = compute_vel_signed(w_vals, vel)
    ax.plot(signed_vel, '-k')

    ax.set_title('(b) Signed Velocity Profile')
    ax.axhline(0, color='r', linestyle=':')
    ax.set_xlabel('Frames', fontsize=12)
    ax.set_ylabel('mm/s', fontsize=12)
    if viz_dist_thresh:
        ax.axvline(w_thresh_indx, color='g', linestyle='-.', label='$d_s$ Distance Threshold')
    ax.axvline(se, c='b', ls='--', label='Segmentation')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f_path, bbox_inches='tight')
    plt.close()

    ####################


def segmentation(x, y, z, vel=None, framerate=None, alpha=0.2, d_threshold=True):
    """
    Automated Targeting Phase Segmentation Algorithm implementation.

    Parameters
    ----------
    x : list or array
        x-axis coordinates.

    y : list or array
        y-axis coordinates.

    z : list or array
        z-axis coordinates.

    alpha : float (default 0.2)
        Parameter used to determine the distance threhsold along displacement d.

    med_filt_w_size : int (default 3)
        Window size for the median filter on the velocity profile prior to segmentation. Depending
        on your dataset, different values for this or an entirely different filtering method
        may be preferred.

    Returns
    -------
    Tuple of (values of d : list, threshold index along d : int, phase segmentation index : int)
    """
    if vel is None:
        # get velocity profile
        vel = compute_velocity(x, y, z, 1/framerate)[0]
    
    # start of trajectory
    start = {
        'x' : x[0],
        'y' : y[0],
        'z' : z[0]
    }
    
    # end of trajectory
    end = {
        'x' : x[-1],
        'y' : y[-1],
        'z' : z[-1]
    }
    
    # projection
    denom = np.sqrt((end['x']-start['x'])**2 + (end['y']-start['y'])**2 + (end['z']-start['z'])**2)
    A = {
        'x' : (end['x']-start['x']) / denom,
        'y' : (end['y']-start['y']) / denom,
        'z' : (end['z']-start['z']) / denom
    }
    
    d_values = []
    for (_x, _y, _z) in zip(x, y, z):
        b_i = {
            'x' : _x - start['x'],
            'y' : _y - start['y'],
            'z' : _z - start['z']
        }
        
        d_i = b_i['x'] * A['x'] + b_i['y'] * A['y'] + b_i['z'] * A['z']
        
        d_values.append(d_i)

    end_point = (end['x'], end['y'], end['z'])
    traj_displacements = [euclidean(i, end_point) for i in zip(x, y, z)]
    d_thresh = traj_displacements[0] * alpha
    d_thresh_indx = [i for i,d in enumerate(traj_displacements) if d <= d_thresh][0]

    # max velocity point
    point_p = np.array([np.argmax(vel), vel[np.argmax(vel)]])
    # point where object grasp/touch occurs
    point_g = np.array([len(vel)-1, vel[-1]])

    # shortest distance to line segment
    shortest_dist = []
    for i, v in enumerate(vel):
        point_i = np.array([i, v])
        cross_product_val = np.cross(point_g-point_p,point_i-point_p)
        d_ni = -1 * cross_product_val
        shortest_dist.append(d_ni)

    if d_threshold:
        # point most distant from line segment connecting velocity peak and end of velocity time series
        # this point is after d_thresh
        segmentation_index = np.nanargmax(shortest_dist[d_thresh_indx:]) + d_thresh_indx
    else:
        # alternatively, search for segmentation after peak velocity magnitude
        segmentation_index = np.nanargmax(shortest_dist[np.argmax(vel):]) + np.argmax(vel)
    
#     return d_values, d_thresh_indx, segmentation_index
    return d_values, d_thresh_indx, segmentation_index, traj_displacements


def compute_velocity(x_coords, y_coords, z_coords, frame_rate=30):
    # computes velocity from position
    x_squared_dist = np.square(np.diff(x_coords,  axis=0))
    y_squared_dist = np.square(np.diff(y_coords,  axis=0))
    z_squared_dist = np.square(np.diff(z_coords,  axis=0))

    return np.sqrt(np.sum(np.dstack((x_squared_dist, y_squared_dist, z_squared_dist)), axis=-1)) / (1/frame_rate)

def find_approx_endpoint(x_col, y_col, z_col, vel=None, c_1=2, c_2=3, use_c2=True, d_threshold=True, alpha=0.2, hz=30):
    """
    Based on method described in Sakai et al 2021, segmentation step (ii)
    """
    # 5th order lowpass butterworth filter with cutoff frequency of 10, sampling at 30 Hz
    sos = butter(5, 10, btype='lowpass', output='sos', fs=hz)
    
    # get velocity profile
    if vel is None:
        vel = compute_velocity(x_col, y_col, z_col, frame_rate=hz)[0]
        vel = sosfiltfilt(sos,vel)

    peak_point = np.array([np.argmax(vel), vel[np.argmax(vel)]])  # t_peak
    t_end = len(vel) - 1  # t_1_end, index of end of trajectory

    approx_endpoint = kin_threshold_and_zerocrossing(x_col=x_col, y_col=y_col, z_col=z_col, vel=vel)
        
    # determine whether segmentation step (iii) can be performed
    if t_end >= approx_endpoint + c_1 * (approx_endpoint - int(peak_point[0])):
        # perform third segmentation
        # not clear why this new index is needed...why not just use rest of trajectory?
        new_end_index = approx_endpoint + c_2 * (approx_endpoint - int(peak_point[0]))
        if use_c2:
            d_values, d_thresh_indx, segmentation_index, traj_displacement = segmentation(x_col[:new_end_index], y_col[:new_end_index], z_col[:new_end_index], vel=vel[:new_end_index], d_threshold=d_threshold, alpha=alpha)
        else:
            d_values, d_thresh_indx, segmentation_index, traj_displacement = segmentation(x_col, y_col, z_col, vel=vel, d_threshold=d_threshold, alpha=alpha)
        return d_values, d_thresh_indx, segmentation_index, new_end_index, approx_endpoint, traj_displacement
    else:
        # dwell length not sufficiently long for segmentation in (iii)
        # approx_endpoint is considered the final segmentation
        return None, None, None, None, approx_endpoint, None
    
def kin_threshold_and_zerocrossing(x_col, y_col, z_col, vel=None):
    """
    Based on method described in Sakai et al 2021, segmentation step (ii)
    """
    # 5th order lowpass butterworth filter with cutoff frequency of 10, sampling at 30 Hz
    sos = butter(5, 10, btype='lowpass', output='sos', fs=30)
    
    # get velocity profile
    if vel is None:
        vel = compute_velocity(x_col, y_col, z_col)[0]
        vel = sosfiltfilt(sos,vel)


    peak_point = np.array([np.argmax(vel), vel[np.argmax(vel)]])  # t_peak
    t_end = len(vel) - 1  # t_1_end, index of end of trajectory

    # ii.a - point after peak velocity where velocity reaches 5% of the peak velocity

    indices = (vel[int(peak_point[0]):] <= 0.05 * peak_point[1]).nonzero()[0]
    if len(indices) == 0:
        # condition ii.a not met
        ii_a_seg = None
    else:
        ii_a_seg = min(indices) + int(peak_point[0])

    # ii.b - point when velocity is less than q% of the peak velocity, 
    #     and the acceleration becomes non-negative for the first time.

    q = 0.20  # per pen point paper

    # get acceleration
    acceleration = np.diff(vel) / (1/30)     # TODO this is specific to HVE
    acceleration = sosfiltfilt(sos, acceleration)
    acceleration = np.insert(acceleration, 0, 0)  # matching up indices

    lessq_mask = vel[int(peak_point[0]):] < q * peak_point[1]
    nonneg_mask = acceleration[int(peak_point[0]):] >= 0
    combined_mask = lessq_mask * nonneg_mask
    indices = combined_mask.nonzero()[0]

    if len(indices) == 0:
        # condition ii.b not met
        ii_b_seg = None
    else:
        ii_b_seg = min(indices) + int(peak_point[0])

    # approx_endpoint is t_h in Sakai 2021
    if ii_a_seg is None and ii_b_seg is None:
        # no approx endpoint found, so set to end index of trajectory
        approx_endpoint = t_end
    elif ii_a_seg is None and ii_b_seg is not None:
        approx_endpoint = ii_b_seg
    elif ii_a_seg is not None and ii_b_seg is None:
        approx_endpoint = ii_a_seg
    else:
        # the earliest point
        approx_endpoint = min(ii_a_seg, ii_b_seg)
        
    return approx_endpoint

def min_jerk_traj_vel(x, y, z, seg_indx):
    """
    Compute straight line min jerk trajectory velocity in 3D (TODO implement so can be 2D as well).
    """
    p1 = np.array([x[0], y[0], z[0]])
    p2 = np.array([x[seg_indx], y[seg_indx], z[seg_indx]])
    total_dist = np.sqrt(np.sum((p2-p1)**2))
    tau = np.array(list(range(seg_indx+1)))/seg_indx
    mjt_vel = total_dist*(30*(tau**4)-60*(tau**3)+30*(tau**2))
    return mjt_vel

def mjt_error(mjt_vel, actual_vel):
    """
    Compute error between normalized min jerk trajectory velocity and actual velocity profiles.
    """
    error = mjt_vel/np.max(mjt_vel) - actual_vel/np.max(actual_vel)
    return error