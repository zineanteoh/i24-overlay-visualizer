import torch.multiprocessing as mp
import socket
import _pickle as pickle

from i24_logger.log_writer         import logger,catch_critical,log_warnings


ctx = mp.get_context('spawn')


import numpy as np
import torch
import os
import time
import pymongo

from src.util.bbox                 import space_nms
from src.util.misc                 import plot_scene,colors,Timer
from i24_configparse               import parse_cfg
from src.track.tracker             import get_Tracker, get_Associator
from src.track.trackstate          import TrackState
from src.detect.pipeline           import get_Pipeline
from src.scene.devicemap           import get_DeviceMap
from src.scene.homography          import HomographyWrapper,Homography
from src.detect.devicebank         import DeviceBank
from src.load.gpu_load_multi       import MCLoader, ManagerClock
from src.db_write                  import WriteWrapper

@log_warnings()
def parse_cfg_wrapper(run_config):
    params = parse_cfg("TRACK_CONFIG_SECTION",
                       cfg_name=run_config, SCHEMA=False)
    return params



@catch_critical()
def checkpoint(tstate,next_target_time,save_file = "working_checkpoint.cpkl"):
    """
    Saves the trackstate and next target_time as a pickled object such that the 
    state of tracker can be reloaded for no loss in tracking progress
    
    :param   tstate - TrackState object
    :param   next_target_time - float
    :return  None
    """
    
    with open(save_file,"wb") as f:
        pickle.dump([next_target_time,tstate],f)
    logger.debug("Checkpointed TrackState object, time:{}s".format(next_target_time))

        
@catch_critical()
def load_checkpoint(target_time,tstate,save_file = "working_checkpoint.cpkl"):
    """
    Loads the trackstate and next target_time from pickled object such that the 
    state of tracker can be reloaded for no loss in tracking progress. Requires 
    input time and tstate such that objects can be naively passed through if no 
    save file exists
    
    :param   tstate - TrackState object
    :param   next_target_time - float
    :return  None
    """
        
    if os.path.exists(save_file):
        with open(save_file,"rb") as f:
            target_time,tstate = pickle.load(f)
        
        logger.debug("Loaded checkpointed TrackState object, time:{}s".format(target_time))
        
    else:
        logger.debug("No checkpoint file exists, starting tracking from max min video timestamp")
        
    return target_time,tstate
        
@catch_critical()
def soft_shutdown(target_time,tstate,cleanup = []):
    logger.warning("Soft Shutdown initiated. Either SIGINT or KeyboardInterrupt recieved")
    checkpoint(tstate,target_time,save_file = "working_checkpoint.cpkl")
    
    for i in range(len(cleanup)-1,-1,-1):
        del cleanup[i]
    
    logger.debug("Soft Shutdown complete. All processes should be terminated")
    raise KeyboardInterrupt()

def main():   
    from i24_logger.log_writer         import logger,catch_critical,log_warnings
    logger.set_name("Tracking Main")
    
    #%% run settings    
    tm = Timer()
    tm.split("Init")
    
    run_config = "execute.config"       
    # mask = ["p46c01", "p46c02", "p46c03", "p46c04", "p46c05", "p46c06"]
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
    tstate = TrackState()
    target_time = None
    
    # load checkpoint
    target_time,tstate = load_checkpoint(target_time,tstate)
    print(target_time)
    
    loader = MCLoader(in_dir, dmap.camera_mapping_file,dmap.cam_names, ctx,start_time = target_time)
    
    logger.debug("Initialized {} loader processes.".format(len(loader.device_loaders)))
    
    #%% more init stuff 
    
    # initialize Homography object
    hg = HomographyWrapper(hg1 = params.eb_homography_file,hg2 = params.wb_homography_file)
     
    if params.track:
        # initialize pipelines
        pipelines = params.pipelines
        pipelines = [get_Pipeline(item, hg) for item in pipelines]
        
        associators = params.associators
        associators = [get_Associator(item) for item in associators]
        
        
        # add Associate function to each pipeline
        # for i in range(len(pipelines)):
        #     assoc = associators[i]
        #     pipelines[i].associate = associators[i]
        
        # initialize DetectorBank
        dbank = DeviceBank(params.cuda_devices, pipelines, dmap.gpu_cam_names, ctx)
        
    # initialize DBWriter object
    if params.write_db:
        dbw = WriteWrapper()
    else:
        dbw = []
    

    
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
    
    # start pymongo instance for plotting
    client = pymongo.MongoClient(host="10.2.218.56",
                                 port=27017,
                                 username="i24-data",
                                 password="mongodb@i24",
                                 connect=True,
                                 connectTimeoutMS=5000)
    db = client["zitest"]
    id_collection = db["batch_5_07072022"]
    transformed_collection = db["batch_5_07072022_transformed"]
    
    # plot first frame
    if params.plot:
        plot_scene(tstate, frames, ts_trunc, dmap.gpu_cam_names,
             hg, colors,extents=dmap.cam_extents_dict, mask=mask,fr_num = frames_processed,detections = None, 
             id_collection=id_collection, transformed_collection=transformed_collection, start_ts=start_ts)
    
    #%% Main Processing Loop
    start_time = time.time()
    
    logger.debug("Initialization Complete. Starting tracking at {}s".format(start_time))
    
    end_time = np.inf
    if params.end_time != -1:
        end_time = params.end_time
        
    # readout headers
    try:
        print("\n\nFrame:    Since Start:  Frame BPS:    Sync Timestamp:     Max ts Deviation:     Active Objects:    Written Objects:")
        while target_time < end_time:
            frames_processed += 1
    
            # optionally, plot outputs
            if params.plot:
                tm.split("Plot")
                detections = None
                priors = None
                plot_scene(tstate, frames, ts_trunc, dmap.gpu_cam_names,
                     hg, colors,extents=dmap.cam_extents_dict, mask=mask,fr_num = frames_processed,detections = detections,priors = priors, 
                     id_collection=id_collection, transformed_collection=transformed_collection, start_ts=start_ts)
            
            # text readout update
            tm.split("Bookkeeping")
            fps = frames_processed/(time.time() - start_time)
            dev = [np.abs(t-target_time) for t in timestamps]
            max_dev = max(dev)
            print("\r{}        {:.3f}s       {:.2f}        {:.3f}              {:.3f}                {}               {}".format(frames_processed, time.time() - start_time,fps,target_time, max_dev, len(tstate), len(dbw)), end='\r', flush=True)
        
            # get next target time
            target_time = clock.tick(timestamps)
            
            # get next frames and timestamps
            tm.split("Get Frames")
            frames, timestamps = loader.get_frames(target_time)
            ts_trunc = [item - start_ts for item in timestamps]
    
            if frames_processed % 20 == 1:
                metrics = {
                    "frame bps": fps,
                    "frame batches processed":frames_processed,
                    "run time":time.time() - start_time,
                    "scene time processed":target_time - start_ts,
                    "active objects":len(tstate),
                    "total terminated objects":term_objects,
                    "avg skipped frames per processed frame": nom_framerate*(target_time - start_ts)/frames_processed -1
                    }
                
            if frames_processed % 100 == 0:
                checkpoint(target_time,tstate)
       
        
        checkpoint(tstate, target_time)
        logger.info("Finished over input time range. Shutting down.")
        
    except KeyboardInterrupt:
        logger.debug("Keyboard Interrupt recieved. Initializing soft shutdown")
        soft_shutdown(target_time, tstate,cleanup = [dbw,loader,dbank])
     
        
     
if __name__ == "__main__":
    main()
