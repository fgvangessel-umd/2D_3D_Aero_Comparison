import numpy as np

def reorder_coords(df_slice, return_indices=False):

    # Now get Connectivities
    Node_C1 = np.array(df_slice['NodeC1'].dropna().values).astype(int) # A list of [1,2,3,4,...]
    Node_C2 = np.array(df_slice['NodeC2'].dropna().values).astype(int) # A list of [2,3,4,5,...]
    Connectivities = np.concatenate((Node_C1.reshape(-1, 1), Node_C2.reshape(-1, 1)), axis=1) # A list of [[1,2],[2,3],[3,4],...]

    # plot the x 'CoordinateX' and y 'CoordinateY' coordinates of the slice
    coords_x = df_slice['CoordinateX'].values
    coords_y = df_slice['CoordinateY'].values

    # We also have XoC YoC ZoC VelocityX VelocityY VelocityZ CoefPressure Mach 
    # We would like to reorder these values in the same way as the coordinates, so we keep track of the indices
    indicies = np.arange(len(df_slice))

    x_deltas = np.zeros(len(Connectivities))
    id_breaks_start_id = [0]
    id_breaks_end_id = []
    prev_id = 0
    segment_ids = np.zeros(len(Connectivities))
    seg_id = 0
    
    for j in range(len(Connectivities)):
        coordx = coords_x[Connectivities[j]-1]
        x_delta = coordx[1] - coordx[0]
        x_deltas[j] = x_delta
        if not Connectivities[j][0]-1 == prev_id:
            # This means that we have a new set of points
            id_breaks_start_id.append(Connectivities[j][0]-1)
            id_breaks_end_id.append(prev_id)
            seg_id += 1
            segment_ids[j] = seg_id
        else:
            segment_ids[j] = seg_id

        prev_id = Connectivities[j][1] - 1

    id_breaks_end_id.append(j)
   
    unique_segment_ids = np.arange(seg_id+1)
    new_seg_order = unique_segment_ids.copy()

    # Loop over and sort the segments such that the end of each x and y coordinate for each segment is the start of the next segment
    # Loop through the segment ids
    seg_coords_start_x = coords_x[id_breaks_start_id]
    seg_coords_start_y = coords_y[id_breaks_start_id]
    seg_coords_end_x = coords_x[id_breaks_end_id]
    seg_coords_end_y = coords_y[id_breaks_end_id]

    err = 1e-8
    for j in range(len(unique_segment_ids)):
        seg_coords_start_x_j = seg_coords_start_x[j]
        seg_coords_start_y_j = seg_coords_start_y[j]
        seg_coords_end_x_j = seg_coords_end_x[j]
        seg_coords_end_y_j = seg_coords_end_y[j]
        # Loop through the segment ids
        seg_x_diff_start = np.abs(seg_coords_start_x_j - seg_coords_end_x)
        seg_y_diff_start = np.abs(seg_coords_start_y_j - seg_coords_end_y)
        seg_tot_diff_start = seg_x_diff_start + seg_y_diff_start
        seg_tot_diff_min_id_start = np.argmin(seg_tot_diff_start)
        seg_tot_diff_min_start = seg_tot_diff_start[seg_tot_diff_min_id_start]
        
        seg_x_diff_end = np.abs(seg_coords_end_x_j - seg_coords_start_x)
        seg_y_diff_end = np.abs(seg_coords_end_y_j - seg_coords_start_y)
        seg_tot_diff_end = seg_x_diff_end + seg_y_diff_end
        seg_tot_diff_min_id_end = np.argmin(seg_tot_diff_end)

        new_seg_order_temp = unique_segment_ids.copy()

        if seg_tot_diff_min_start > err:
            # No segment matches, so we have found the starting segment
            # Now we need to reorder the segments
            # Put the first segment in the first position
            new_seg_order_temp[0] = unique_segment_ids[j]
            new_seg_order_temp[j] = unique_segment_ids[0]
            # Now choose the id that is minimum; i.e we match the end of the first segment to the start of the next most likely segment
            new_seg_order_temp[1] = unique_segment_ids[seg_tot_diff_min_id_end]
            new_seg_order_temp[seg_tot_diff_min_id_end] = unique_segment_ids[1]
            # Now loop through the remaining segments and match the end of the previous segment to the start of the next most likely segment, breaking if we reach the end
            for k in range(2,len(unique_segment_ids)):
                # Get the previous segment id
                prev_seg_id = new_seg_order_temp[k-1]
                # Get the previous segment end coordinates
                prev_seg_end_x = seg_coords_end_x[prev_seg_id]
                prev_seg_end_y = seg_coords_end_y[prev_seg_id]
                # Get the remaining segment ids
                remaining_seg_ids = np.setdiff1d(unique_segment_ids, new_seg_order_temp[:k])
                # Get the remaining segment start coordinates
                remaining_seg_start_x = seg_coords_start_x[remaining_seg_ids]
                remaining_seg_start_y = seg_coords_start_y[remaining_seg_ids]
                # Get the difference between the previous segment end coordinates and the remaining segment start coordinates
                remaining_seg_x_diff = np.abs(prev_seg_end_x - remaining_seg_start_x)
                remaining_seg_y_diff = np.abs(prev_seg_end_y - remaining_seg_start_y)
                remaining_seg_tot_diff = remaining_seg_x_diff + remaining_seg_y_diff
                # Get the minimum difference
                remaining_seg_tot_diff_min_id = np.argmin(remaining_seg_tot_diff)
                # Now get the id of the remaining segment that matches the previous segment end coordinates
                remaining_seg_id = remaining_seg_ids[remaining_seg_tot_diff_min_id]
                # Now put the remaining segment id in the next position
                new_seg_order_temp[k] = remaining_seg_id
            
            # Now we have a new segment order
            new_seg_order = new_seg_order_temp
            break

    # We can use the new order to plot all segments at once
    # Concatenate the segments in the new order
    coords_x_reordered = np.array([])
    coords_y_reordered = np.array([])
    indicies_reordered = np.array([])

    for j in range(len(new_seg_order)):
        segment = np.nonzero(segment_ids == new_seg_order[j])[0]
        # print('seg shape',segment.shape)
        coords_x_segment = coords_x[Connectivities[segment]-1][:,0]
        coords_y_segment = coords_y[Connectivities[segment]-1][:,0]
        indicies_segment = indicies[Connectivities[segment]-1][:,0]
        coords_x_reordered = np.concatenate((coords_x_reordered, coords_x_segment))
        coords_y_reordered = np.concatenate((coords_y_reordered, coords_y_segment))
        indicies_reordered = np.concatenate((indicies_reordered, indicies_segment))
    
    if return_indices:
        return indicies_reordered.astype(int)
    else:
        return coords_x_reordered, coords_y_reordered
