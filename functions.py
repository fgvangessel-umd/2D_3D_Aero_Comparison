import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
import glob
from reorder import reorder_coords

import numpy as np
from scipy.interpolate import splprep, splev, BSpline, spalde

def read_2d_slice_data(fname):
    # Read in 2D data airfoil data from tecplot formatted file named in fname
    file_path = fname
    with open(file_path, 'r') as f:
        s = f.readlines()
    
    # Extract 2D surface data
    for i, line in enumerate(s):
        if line.strip().startswith('DATAPACKING=POINT'):
            nnod = int(s[i-1].split('=')[1].split()[0].split(',')[0])
            nelm = int(s[i-1].split('=')[2].split()[0].split(',')[0])
    
            d2_dat = s[i+1:i+1+nnod]
            
            d2_dat = np.genfromtxt(d2_dat, dtype=float)
    
            break
    
    d2_df = pd.DataFrame(d2_dat[:,[0,1,2,6,7,8,9]], columns=['CoordinateX', 'CoordinateY', 'CoordinateZ', 'VX', 'VY', 'VZ', 'CP'])
    
    connectivity = np.genfromtxt(s[i+1+nnod:], dtype=float)
    df_conn = pd.DataFrame(connectivity, columns=['NodeC1', 'NodeC2'])
    
    result = pd.concat([d2_df, df_conn], axis=1)
    d2_df = result.reindex(d2_df.index)
    
    index_list = list(reorder_coords(d2_df, return_indices=True))
    d2_df = d2_df.loc[index_list]

    return d2_df

def read_3d_slice_data(fname):
    '''
    Take in file containing 3D slice data and return dataframes of the data for each slice as a list
    '''
    with open(fname, 'r') as f:
        s = f.readlines()
        
    # Find the lines where the zone data starts and ends
    start = []
    nzones=0

    nnods = []
    nelms = []
    data = []
    zcoords = []

    for i, line in enumerate(s):
        if line.strip().startswith('DATAPACKING=POINT'):
            start.append(i+1)
            nnod = int(s[i-1].split('=')[1].split()[0])
            nelm = int(s[i-1].split('=')[2].split()[0].split(',')[0])
            nnods.append(nnod)
            nelms.append(nelm)
            nzones += 1

            # Get zcoordinates
            z = float(s[i-2].split('=')[-1].split()[0][:-1])
            zcoords.append(z)

        else:
            continue

    for ind, nnod, nelm, zcoord in zip(start,nnods, nelms, zcoords):
        slice_dat = np.genfromtxt(s[ind:ind+nnod], dtype=float)
        df = pd.DataFrame(slice_dat[:,[0,1,2,6,7,8,9]], columns=['CoordinateX', 'CoordinateY', 'CoordinateZ', 'VX', 'VY', 'VZ', 'CP'])
        connectivity = np.genfromtxt(s[ind+nnod:ind+nnod+nelm], dtype=float)
        df_conn = pd.DataFrame(connectivity, columns=['NodeC1', 'NodeC2'])
        result = pd.concat([df, df_conn], axis=1)
        df = result.reindex(df.index)
        index_list = list(reorder_coords(df, return_indices=True))
        df = df.loc[index_list]

        df['zcoord'] = zcoord
        
        data.append(df)

        
    return data

def closest_point(x, y, xtarget, ytarget):
    '''
    Identify closest (Euclidean) point in x, y coordinate list to a single point (xtarget, ytarget)
    '''
    xy = np.concatenate((x.reshape(-1,1), y.reshape(-1,1)), axis=1)
    xy_target = np.ones(xy.shape) @ np.diag([xtarget, ytarget])

    d = np.sqrt(np.sum((xy - xy_target)**2, axis=1))

    return np.argmin(d)

def two_closest_points(x, y, xtarget, ytarget):
    '''
    Identify two closest (Euclidean) point in x, y coordinate list to a single point (xtarget, ytarget)
    '''
    xy = np.concatenate((x.reshape(-1,1), y.reshape(-1,1)), axis=1)
    xy_target = np.ones(xy.shape) @ np.diag([xtarget, ytarget])
    
    d = np.sqrt(np.sum((xy - xy_target)**2, axis=1))

    idx = np.argsort(d)[:2]

    xy1 = xy[idx[0],:]
    xy2 = xy[idx[1],:]

    d1 = d[idx[0]]
    d2 = d[idx[1]]

    return xy1, xy2, d1, d2, idx[0], idx[1]

def fit_raw_Bspline(x, y):
    ''' Fit parameterized B-Spline to curve characterized by list of points (x,y) and find point of maximum curvature '''
    tck, u = splprep([x, y], s=0, k=3)

    # Assumption: s=0.7 encompasses point of maximum curvature
    u = np.linspace(0., 1.0, 1000)
    new_points = splev(u, tck, der=0)
    xspline, yspline = new_points[0], new_points[1]

    d0, d1 = splev(u, tck, der=3)
    idxs = np.argsort(np.abs(d1))[::-1]
    imax = idxs[0]

    xtip, ytip = xspline[imax], yspline[imax]

    # Fit entire wing surface
    s = np.linspace(0., 1.0, 1000)
    new_points = splev(s, tck)
    xspline, yspline = new_points[0], new_points[1]

    # Get coordinate index closest to B-spline tip
    imin = closest_point(x, y, xtip, ytip)
    xtip, ytip = x[imin], y[imin]
    idx_tip = imin

    return xspline, yspline, s, xtip, ytip, idx_tip

def rotate_scale(x, y, xtip, ytip):
    '''
    Rotate and scale airfoil coordinate points such that leading edge is located at (0,0) and
    trailing tip is located at (1,0)
    '''
     # Rotate
    t = np.arctan(ytip/xtip)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    xy = np.concatenate((x.reshape(-1,1), y.reshape(-1,1)), axis=1)
    xy = xy @ R
    x, y = xy[:,0], xy[:,1]
    xytip = np.array([[xtip, ytip]]) @ R
    xtip, ytip = xytip[0,0], xytip[0,1]

    # Scale volume equally along x and y axes
    V = np.array([[1/xtip, 0],[0, 1/xtip]])
    xy = np.concatenate((x.reshape(-1,1), y.reshape(-1,1)), axis=1)
    xy = xy @ V
    x, y = xy[:,0], xy[:,1]
    xytip = np.array([[xtip, ytip]]) @ V
    xtip, ytip = xytip[0,0], xytip[0,1]

    return x, y, xtip, ytip

def fit_processed_Bspline(x, y, ns=int(1e3)):
    ''' 
    Fit B-Spline to processed data defined by list of x, y coordinate points. 
    Evenly sample ns points along parameterized spline curve
    '''
    tck, _ = splprep([x, y], s=0, k=3)

    # Fit entire wing surface
    s = np.linspace(0., 1.0, ns)
    new_points = splev(s, tck)
    xspline, yspline = new_points[0], new_points[1]

    return xspline, yspline, s, tck

def separate_top_bottom(x, y, vars, idx_tip):
    '''
    Separate data (geometric and/or fluid) into top and bottom components
    '''

    if len(vars.shape) == 2:
        vars.reshape((-1,1))

    xb, yb, vb = x[:idx_tip], y[:idx_tip], vars[:idx_tip,:]
    xt, yt, vt = x[idx_tip:], y[idx_tip:], vars[idx_tip:,:]
    xyb = np.concatenate((xb.reshape(-1,1), yb.reshape(-1,1)), axis=1)
    xyt = np.concatenate((xt.reshape(-1,1), yt.reshape(-1,1)), axis=1)
    idx_tip -= 1

    return xyb, xyt, vb, vt, idx_tip

def spline_field_fit(xy, var, ns=int(1e3)):
    '''
    Fit field variables along airfoil surface (generally one would call this either on the bottom or the top surface)
    '''

    # Fit B-Spline to bottom processed coordinate data
    xs, ys, s, tck = fit_processed_Bspline(xy[:,0], xy[:,1], ns)
    xys = np.concatenate((xs.reshape(-1,1), ys.reshape(-1,1)), axis=1)
    
    # Spline fit field (either bottom or top)
    vs = np.zeros((ns, var.shape[1]))
    for i, (xx, yy) in enumerate(zip(xs, ys)):
        xy1, xy2, d1, d2, i1, i2 = two_closest_points(xy[:,0], xy[:,1], xx, yy)
        #d = d1+d2
        #w1, w2 = 1-d1/d, 1-d2/d
        #vs[i] = w1*var[i1] + w2*var[i2]
        vs[i,:] = var[i1,:] + (var[i2,:] - var[i1,:])/(xy2[0] - xy1[0])*(xx-xy1[0])

    return xys, vs, s

def parameterize_airfoil_curve(df, ns=100):
    '''
    Stitch together multiple fundamental functions to parameterize airfoils using B-splines.
    Paramterize entire airfoil, as well as top and bottom separately
    '''
    # Fit raw data to B-Splines
    x, y = df['CoordinateX'].values, df['CoordinateY'].values
    xspline, yspline, s, xtip, ytip, idx_tip = fit_raw_Bspline(df)
    x, y, xtip, ytip = rotate_scale(x, y, xtip, ytip)

    # Fit B-Spline to processed coordinate data
    xspline, yspline, s, tck = fit_processed_Bspline(x, y)
    xyspline = np.concatenate((xspline.reshape(-1,1), yspline.reshape(-1,1)), axis=1)

    # Get spline interpolation point falling closest to trailing tip of original data
    i = closest_point(xspline, yspline, xtip, ytip)
    midpoint = splev(np.array(s[i]), tck)
    xy = np.concatenate((x.reshape(-1,1), y.reshape(-1,1)), axis=1)
    
    # sample spline along top and bottom
    sbottom = np.linspace(0, s[i], ns)
    stop = np.linspace(s[i], 1., ns)
    xspline_b, yspline_b = splev(sbottom, tck)
    xspline_t, yspline_t = splev(stop, tck)
    
    xyspline_b = np.concatenate((xspline_b.reshape(-1,1), yspline_b.reshape(-1,1)), axis=1)
    xyspline_t = np.concatenate((xspline_t.reshape(-1,1), yspline_t.reshape(-1,1)), axis=1)

    return xyspline, xyspline_b, xyspline_t, s, sbottom, stop, midpoint, xy, xtip, ytip

def calculate_geom_distance(df1, df2):
    '''
    Calcualte distance between two airfoil geometries by summing the pointwise Euclidesan distance between 
    the (x,y) locations at a common parameterization length s
    '''
    _, xys_b_1, xys_t_1, _, _, _, _, _, _, _ = parameterize_airfoil_curve(df1)
    _, xys_b_2, xys_t_2, _, _, _, _, _, _, _ = parameterize_airfoil_curve(df2)
    
    dist_b = np.sum(np.sqrt(np.sum((xys_b_1 - xys_b_2)**2, axis=1)))
    dist_t = np.sum(np.sqrt(np.sum((xys_t_1 - xys_t_2)**2, axis=1)))
    dist = dist_b + dist_t

    return dist