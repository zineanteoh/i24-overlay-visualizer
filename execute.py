import torch.multiprocessing as mp
import numpy as np
import torch
import time
import pymongo
from i24_logger.log_writer         import logger, catch_critical, log_warnings
from src.scene.homography          import HomographyWrapper, Homography
from src.load.gpu_load_multi       import MCLoader, ManagerClock
from src.util.misc                 import plot_scene,colors,Timer
from src.scene.devicemap           import get_DeviceMap
from i24_configparse               import parse_cfg

ctx = mp.get_context('spawn')

@catch_critical()
def soft_shutdown(target_time,cleanup = []):
    for i in range(len(cleanup)-1,-1,-1):
        del cleanup[i]
    
    logger.debug("Soft Shutdown complete. All processes should be terminated")
    raise KeyboardInterrupt()

def main(): 
    
    # start pymongo instance for plotting
    client = pymongo.MongoClient(host="10.2.218.56",
                                 port=27017,
                                 username="i24-data",
                                 password="mongodb@i24",
                                 connect=True,
                                 connectTimeoutMS=5000)
    
    # select collections to query from
    db = client["zitest"]
    id_collection = db["batch_5_07072022"]
    transformed_collection = db["batch_5_07072022_transformed"]
    
    # determine if collection is RAW or RECONCILED
    first_doc = transformed_collection.find_one()
    if 'dimensions' in first_doc:
        MODE = "RAW"
    else:
        MODE = "RECONCILED"
    
    # set which cameras to plot
    # ... to plot all cameras: mask = None 
    # ... to plot a list of cameras: mask = ["p46c01", "p46c02", "p46c03", "p46c04", "p46c05", "p46c06"]
    mask = None
    
    #%% run settings
    logger.set_name("Overlay Bounding Box Visualizer")
    
    tm = Timer()
    tm.split("Init")
    
    # load parameters from config and load in_dir, the directory of the video files
    run_config = "execute.config"
    params = parse_cfg("DEFAULT", cfg_name=run_config, SCHEMA=False)
    in_dir = params.input_directory
    
    # TODO fix this once you redo configparse
    params.cuda_devices = [int(i) for i in range(int(params.cuda_devices))]
    assert max(params.cuda_devices) < torch.cuda.device_count()
    
    # intialize DeviceMap
    dmap = get_DeviceMap(params.device_map)
    loader = MCLoader(in_dir, dmap.camera_mapping_file,dmap.cam_names, ctx,start_time = None)
    logger.debug("Initialized {} loader processes.".format(len(loader.device_loaders)))
    
    # initialize Homography object
    hg = HomographyWrapper(hg1 = params.eb_homography_file,hg2 = params.wb_homography_file)

    # get frames and timestamps
    frames, timestamps = loader.get_frames(target_time = None)
    
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
    
    #%% 
    # plot first frame
    if params.plot:
        plot_scene(MODE, frames, ts_trunc, dmap.gpu_cam_names,
             hg, colors,extents=dmap.cam_extents_dict, mask=mask,fr_num = frames_processed, 
             id_collection=id_collection, transformed_collection=transformed_collection, start_ts=start_ts)
    
    #%% Main Processing Loop
    start_time = time.time()
    
    logger.debug("Initialization Complete. Starting overlay visualizer at {}s".format(start_time))
    
    end_time = np.inf
    if params.end_time != -1:
        end_time = params.end_time
    
    # readout headers
    try:
        print("\n\nFrame:     Since Start:     Frame BPS:     Sync Timestamp:     Max ts Deviation:")
        while target_time < end_time:
            frames_processed += 1
            
            # plot overlay bounding boxes
            if params.plot:
                tm.split("Plot")
                plot_scene(MODE, frames, ts_trunc, dmap.gpu_cam_names,
                     hg, colors,extents=dmap.cam_extents_dict, mask=mask,fr_num = frames_processed, 
                     id_collection=id_collection, transformed_collection=transformed_collection, start_ts=start_ts)
            
            # text readout update
            tm.split("Bookkeeping")
            fps = frames_processed/(time.time() - start_time)
            dev = [np.abs(t-target_time) for t in timestamps]
            max_dev = max(dev)
            print("{}     {:.3f}s     {:.2f}     {:.3f}     {:.3f}".format(frames_processed, time.time() - start_time,fps,target_time, max_dev))
            
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
                    "avg skipped frames per processed frame": nom_framerate*(target_time - start_ts)/frames_processed -1
                    }
                print(metrics)
        
        logger.info("Finished over input time range. Shutting down.")
        
    except KeyboardInterrupt:
        logger.debug("Keyboard Interrupt recieved. Initializing soft shutdown")
        soft_shutdown(target_time, cleanup = [loader])

if __name__ == "__main__":
    main()
