from torchvision import transforms
import torch
import numpy as np
import cv2
import time

colors = np.random.randint(0,255,[1000,3])
colors[:,0] = 0.2

class Timer:
    
    def __init__(self):
        self.cur_section = None
        self.sections = {}
        self.section_calls = {}
        
        self.start_time= time.time()
        self.split_time = None
        
    def split(self,section,SYNC = False):

        # store split time up until now in previously active section (cur_section)
        if self.split_time is not None:
            if SYNC:
                torch.cuda.synchronize()
                
            elapsed = time.time() - self.split_time
            if self.cur_section in self.sections.keys():
                self.sections[self.cur_section] += elapsed
                self.section_calls[self.cur_section] += 1

            else:
                self.sections[self.cur_section] = elapsed
                self.section_calls[self.cur_section] = 1
        # start new split and activate current time
        self.cur_section = section
        self.split_time = time.time()
        
    def bins(self):
        return self.sections
    
    def __repr__(self):
        out = ["{}:{:2f}s/call".format(key,self.sections[key]/self.section_calls[key]) for key in self.sections.keys()]
        return str(out)

def transform_docs_to_boxes(timestamp, id_collection, transformed_collection):
    """
    
    """
    
    # query an upper timestamp and lower timestamp
    lower = (np.floor(timestamp * 25) / 25).item()
    upper = (np.ceil(timestamp * 25) / 25).item()
    lower_timestamp_doc = transformed_collection.find_one({"timestamp": lower})
    upper_timestamp_doc = transformed_collection.find_one({"timestamp": upper})
    
    print("Camera timestamp: {}. Interpolating between {} to {}".format(timestamp, lower, upper))
    
    # query for all trajectory documents whose index is in the union of the two timestamp docs
    # ... apply projectiong to only return necessary fields
    lower_id_set = set(lower_timestamp_doc["id"])
    upper_id_set = set(upper_timestamp_doc["id"])
    traj_cursor = id_collection.find({"_id": {"$in": list(lower_id_set | upper_id_set)}}, 
                                    {"width":1, "length":1, "height":1, "direction":1, "coarse_vehicle_class":1})
    
    box = []
    classes = []
    vehicle_ids = []
    for index, traj in enumerate(traj_cursor):
        # traj:
        # {
        #     timestamp: 0.0
        #     id: [a, b, c, ...]
        #     x_position: [1, 2, 3, ...],
        #     y_position: [1, 2, 3, ...]
        # }
        vehicle_id = traj["_id"]
        vehicle_ids.append(vehicle_id)
        
        if vehicle_id in lower_id_set.intersection(upper_id_set):
            # interpolate x and y position
            lower_index = lower_timestamp_doc["id"].index(vehicle_id)
            upper_index = upper_timestamp_doc["id"].index(vehicle_id)
            lower_x = lower_timestamp_doc["x_position"][lower_index]
            upper_x = upper_timestamp_doc["x_position"][upper_index]
            lower_y = lower_timestamp_doc["y_position"][lower_index]
            upper_y = upper_timestamp_doc["y_position"][upper_index]
            x_position = np.interp(timestamp, [lower, upper], [lower_x, upper_x]).item()
            y_position = np.interp(timestamp, [lower, upper], [lower_y, upper_y]).item()
        elif vehicle_id in lower_id_set.difference(upper_id_set):
            # plot x,y from lower timestamp
            lower_index = lower_timestamp_doc["id"].index(vehicle_id)
            x_position = lower_timestamp_doc["x_position"][lower_index]
            y_position = lower_timestamp_doc["y_position"][lower_index]
        else:
            # plot x,y from upper timestamp
            upper_index = upper_timestamp_doc["id"].index(vehicle_id)
            x_position = upper_timestamp_doc["x_position"][upper_index]
            y_position = upper_timestamp_doc["y_position"][upper_index]
        
        direction = traj["direction"]
        
        length = np.median(traj["length"]).item()
        width = np.median(traj["width"]).item()
        height = np.median(traj["height"]).item()
        
        classes.append(traj["coarse_vehicle_class"])
        box.append([x_position, y_position, length, width, height, direction, 0])
    box = torch.tensor(box)
    classes = torch.tensor(classes)
    return classes, box, vehicle_ids


def plot_scene(frames, ts, gpu_cam_names, hg, colors, mask=None, extents=None, fr_num = 0, id_collection=None, transformed_collection=None, start_ts=None):
    """
    Plots the set of active cameras, or a subset thereof
    ts     - stack of camera timestamps
    frames - stack of frames as pytorch tensors
    hg     - Homography Wrapper object
    mask   - None or list of camera names to be plotted
    extents - None or cam extents from dmap.cam_extents_dict
    X_collection - pymongo collection
    start_ts - the timestamp offset of camera (need it for querying timestamp from collection)
    """

    # Internal settings that shouldn't change after initial tuning
    PLOT_TOLERANCE = 50  # feet
    MONITOR_SIZE = (2160, 3840)
    denorm = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                  std=[1/0.229, 1/0.224, 1/0.225])
    #mask = ["p1c1"]

    # 1. prep frames
    # move to CPU
    frames = [item.cpu() for item in frames]

    # stack all frames into single list
    frames = torch.cat(frames, dim=0)
    #ts =  torch.cat(ts,dim = 0)
    cam_names = [item for sublist in gpu_cam_names for item in sublist]

    # get mask
    if mask is not None:
        keep = []
        keep_cam_names = []
        for idx, cam in enumerate(cam_names):
            if cam in mask:
                keep.append(idx)
                keep_cam_names.append(cam)

        # mask relevant cameras
        cam_names = keep_cam_names
        ts = [ts[idx] for idx in keep]
        frames = frames[keep, ...]
    
    # 2. plot boxes
    # for each frame
    plot_frames = []
    for f_idx in range(len(frames)):

        timestamp = ts[f_idx] + start_ts
        
        # get the reported position of each object from mongodb for timestamp 
        classes, boxes, vehicle_ids = transform_docs_to_boxes(timestamp, id_collection, transformed_collection)
        ids = torch.tensor([x for x in range(len(boxes))])
        
        # if extents is not None and len(boxes) > 0:
        #     xmin, xmax, _ = extents[cam_names[f_idx]]

        #     # select objects that fall within that camera's space range (+ some tolerance)
        #     keep_obj = torch.mul(torch.where(boxes[:, 0] > xmin - PLOT_TOLERANCE, 1, 0), torch.where(
        #         boxes[:, 0] < xmax + PLOT_TOLERANCE, 1, 0)).nonzero().squeeze(1)
        #     boxes = boxes[keep_obj,:]
        #     ids = ids[keep_obj]
        #     classes = classes[keep_obj]
        #     classes = [hg.hg1.class_dict[cls.item()] for cls in classes]
                            
        
        # convert frame into cv2 image
        fr = (denorm(frames[f_idx]).numpy().transpose(1, 2, 0)*255)[:,:,::-1]
        #fr = frames[f_idx].numpy().transpose(1,2,0)
        # use hg to plot the boxes and IDs in that camera
        if boxes is not None and len(boxes) > 0:
            
            labels = ["{}: {}".format(classes[i],str(vehicle_ids[i])[-6:]) for i in range(len(ids))]
            color_slice = colors[ids%colors.shape[0],:]
            #color_slice = [colors[id,:] for id in ids]
            #color_slice = np.stack(color_slice)
            if color_slice.ndim == 1:
                 color_slice = color_slice[np.newaxis,:]
            
            fr = hg.plot_state_boxes(
                fr.copy(), boxes, name=cam_names[f_idx], labels=labels,thickness = 3, color = color_slice)

        # plot original detections
        # if detections is not None:
        #     keep_det= torch.mul(torch.where(detections[:, 0] > xmin - PLOT_TOLERANCE, 1, 0), torch.where(
        #         detections[:, 0] < xmax + PLOT_TOLERANCE, 1, 0)).nonzero().squeeze(1)
        #     detections_selected = detections[keep_det,:]        # ids, boxes = tstate(ts[f_idx],with_direction=True)
        # classes = torch.tensor([class_by_id[id.item()] for id in ids])
        # print(boxes.shape)
        # print(classes.shape)
        # print(ids.shape)
        # print(boxes)
        # print(classes)
        # print(ids)
            
        #     fr = hg.plot_state_boxes(
        #         fr.copy(), detections_selected, name=cam_names[f_idx], labels=None,thickness = 1, color = (255,0,0))

        # plot priors
        # if priors is not None and len(priors) > 0:
        #     keep_pr = torch.mul(torch.where(priors[:, 0] > xmin - PLOT_TOLERANCE, 1, 0), torch.where(
        #         priors[:, 0] < xmax + PLOT_TOLERANCE, 1, 0)).nonzero().squeeze(1)
        #     priors_selected = priors[keep_pr,:]
            
        #     fr = hg.plot_state_boxes(
        #         fr.copy(), priors_selected, name=cam_names[f_idx], labels=None,thickness = 1, color = (0,0,255))


        # plot timestamp
        fr = cv2.putText(fr.copy(), "Timestamp: {:.3f}s".format(ts[f_idx]), (10,70), cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),3)
        fr = cv2.putText(fr.copy(), "Camera: {}".format(cam_names[f_idx]), (10,30), cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),3)
        
        # append to array of frames
        plot_frames.append(fr)

    # 3. tile frames
    n_ims = len(plot_frames)
    n_row = int(np.round(np.sqrt(n_ims)))
    n_col = int(np.ceil(n_ims/n_row))

    cat_im = np.zeros([1080*n_row, 1920*n_col, 3])
    for im_idx, original_im in enumerate(plot_frames):
        row = im_idx // n_col
        col = im_idx % n_col
        cat_im[row*1080:(row+1)*1080, col*1920:(col+1)*1920, :] = original_im

    # resize to fit on standard monitor
    trunc_h = cat_im.shape[0] / MONITOR_SIZE[0]
    trunc_w = cat_im.shape[1] / MONITOR_SIZE[1]
    trunc = max(trunc_h, trunc_w)
    new_size = (int(cat_im.shape[1]//trunc), int(cat_im.shape[0]//trunc))
    cat_im = cv2.resize(cat_im, new_size) / 255.0

    cv2.imwrite("/home/derek/Desktop/video_viz/cleaning_code/{}.png".format(str(fr_num).zfill(4)),cat_im*255)
    # plot
    cv2.imshow("frame", cat_im)
    # cv2.setWindowTitle("frame",str(self.frame_num))
    key = cv2.waitKey(1)
    if key == ord("p"):
        cv2.waitKey(0)
    elif key == ord("q"):
        cv2.destroyAllWindows()
        shutdown()

def shutdown():
    raise KeyboardInterrupt("Manual Shutdown triggered")