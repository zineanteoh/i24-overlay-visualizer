#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 11:41:06 2022

@author: Yanbing
Build a video overlay viz once and for all

At least draw a dot for each detection, interpolate the position based on frametime
input: collection, video files
"""
from torchvision import transforms
import torch
import numpy as np
import cv2
import torch.multiprocessing as mp
from i24_database_api.db_reader import DBReader


ctx = mp.get_context('spawn')

from src.util.misc                 import plot_scene,colors,Timer
from i24_configparse               import parse_cfg
from src.track.tracker             import get_Tracker, get_Associator
from src.track.trackstate          import TrackState
from src.detect.pipeline           import get_Pipeline
from src.scene.devicemap           import get_DeviceMap
from src.scene.homography          import HomographyWrapper,Homography
from src.load.gpu_load_multi       import MCLoader, ManagerClock


colors = np.random.randint(0,255,[1000,3])
colors[:,0] = 0.2


# def plot_scene(tstate, frames, ts, gpu_cam_names, hg, colors, mask=None, extents=None, fr_num = 0,detections = None,priors = None):
def plot_scene(col, frames, ts, gpu_cam_names, hg, colors, mask=None, extents=None, fr_num = 0,detections = None,priors = None):
    """
    Plots the set of active cameras, or a subset thereof
    tstate - TrackState object
    col    - pymongo collection object
    ts     - stack of camera timestamps
    frames - stack of frames as pytorch tensors
    hg     - Homography Wrapper object
    mask   - None or list of camera names to be plotted
    extents - None or cam extents from dmap.cam_extents_dict
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

    # class_by_id = tstate.get_classes()
    class_by_id = col.s
    
    # 2. plot boxes
    # for each frame
    plot_frames = []
    for f_idx in range(len(frames)):

        # get the reported position of each object from tstate for that time
        ids, boxes = tstate(ts[f_idx],with_direction=True)
        classes = torch.tensor([class_by_id[id.item()] for id in ids])
        
        
        if extents is not None and len(boxes) > 0:
            xmin, xmax, _ = extents[cam_names[f_idx]]

            # select objects that fall within that camera's space range (+ some tolerance)
            keep_obj = torch.mul(torch.where(boxes[:, 0] > xmin - PLOT_TOLERANCE, 1, 0), torch.where(
                boxes[:, 0] < xmax + PLOT_TOLERANCE, 1, 0)).nonzero().squeeze(1)
            boxes = boxes[keep_obj,:]
            ids = ids[keep_obj]
            classes = classes[keep_obj]
            classes = [hg.hg1.class_dict[cls.item()] for cls in classes]
                            
            
        # convert frame into cv2 image
        fr = (denorm(frames[f_idx]).numpy().transpose(1, 2, 0)*255)[:,:,::-1]
        #fr = frames[f_idx].numpy().transpose(1,2,0)
        # use hg to plot the boxes and IDs in that camera
        if boxes is not None and len(boxes) > 0:
            
            labels = ["{}: {}".format(classes[i],ids[i]) for i in range(len(ids))]
            color_slice = colors[ids%colors.shape[0],:]
            #color_slice = [colors[id,:] for id in ids]
            #color_slice = np.stack(color_slice)
            if color_slice.ndim == 1:
                 color_slice = color_slice[np.newaxis,:]
            
            fr = hg.plot_state_boxes(
                fr.copy(), boxes, name=cam_names[f_idx], labels=labels,thickness = 3, color = color_slice)

        # plot original detections
        if detections is not None:
            keep_det= torch.mul(torch.where(detections[:, 0] > xmin - PLOT_TOLERANCE, 1, 0), torch.where(
                detections[:, 0] < xmax + PLOT_TOLERANCE, 1, 0)).nonzero().squeeze(1)
            detections_selected = detections[keep_det,:]
            
            fr = hg.plot_state_boxes(
                fr.copy(), detections_selected, name=cam_names[f_idx], labels=None,thickness = 1, color = (255,0,0))

        # plot priors
        if priors is not None and len(priors) > 0:
            keep_pr = torch.mul(torch.where(priors[:, 0] > xmin - PLOT_TOLERANCE, 1, 0), torch.where(
                priors[:, 0] < xmax + PLOT_TOLERANCE, 1, 0)).nonzero().squeeze(1)
            priors_selected = priors[keep_pr,:]
            
            fr = hg.plot_state_boxes(
                fr.copy(), priors_selected, name=cam_names[f_idx], labels=None,thickness = 1, color = (0,0,255))


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

    cv2.imwrite("/home/derek/Desktop/temp_frames/{}.png".format(str(fr_num).zfill(4)),cat_im*255)
    # plot
    cv2.imshow("frame", cat_im)
    # cv2.setWindowTitle("frame",str(self.frame_num))
    key = cv2.waitKey(1)
    
    
if __name__ == "__main__":

    
    def parse_cfg_wrapper(run_config):
        params = parse_cfg("TRACK_CONFIG_SECTION",
                           cfg_name=run_config, SCHEMA=False)
        return params
    
    tm = Timer()
    tm.split("Init")
    
    run_config = "execute.config"       
    #mask = ["p46c01","p46c02", "p46c03", "p46c04", "p46c05","p46c06"]
    mask = None
    
    # load parameters
    params = parse_cfg_wrapper(run_config)
    
    in_dir = params.input_directory
    
    
    # TODO fix this once you redo configparse
    params.cuda_devices = [int(i) for i in range(int(params.cuda_devices))]
    assert max(params.cuda_devices) < torch.cuda.device_count()
    
    # intialize DeviceMap
    dmap = get_DeviceMap(params.device_map)
    
    # intialize empty TrackState Object
    # tstate = TrackState()
    target_time = None
    
    # load checkpoint
    # target_time,tstate = load_checkpoint(target_time,tstate)
    loader = MCLoader(in_dir, dmap.camera_mapping_file,dmap.cam_names, ctx,start_time = target_time)
    
    
    # initialize Homography object
    hg = HomographyWrapper(hg1 = params.eb_homography_file,hg2 = params.wb_homography_file)
    
    # get frames and timestamps
    frames, timestamps = loader.get_frames(target_time = target_time)

    # initialize processing sync clock
    start_ts = max(timestamps)
    nom_framerate = params.nominal_framerate 
    clock  = ManagerClock(start_ts,params.desired_processing_speed, nom_framerate)
    target_time = start_ts


    # initial sync-up of all cameras
    # TODO - note this means we always skip at least one frame at the beginning of execution
    frames,timestamps = loader.get_frames(target_time)
    ts_trunc = [item - start_ts for item in timestamps]

    frames_processed = 0
    term_objects = 0

    
        
    # define collection
    
        
        
    plot_scene(col, frames, ts_trunc, dmap.gpu_cam_names,
         hg, colors,extents=dmap.cam_extents_dict, mask=mask,fr_num = frames_processed,detections = None)