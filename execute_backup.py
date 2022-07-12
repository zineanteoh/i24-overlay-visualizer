import torch.multiprocessing as mp
import socket
import _pickle as pickle

from i24_logger.log_writer         import logger,catch_critical,log_warnings


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
    logger.debug("Checkpointed TrackState object, time:{}s".format(target_time))

        
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
        
@catch_critical()
def soft_shutdown(target_time,tstate,cleanup = []):
    logger.warning("Soft Shutdown initiated. Either SIGINT or KeyboardInterrupt recieved")
    checkpoint(tstate,target_time,save_file = "working_checkpoint.cpkl")
    
    for i in range(len(cleanup)-1,-1,-1):
        del cleanup[i]
    
    logger.debug("Soft Shutdown complete. All processes should be terminated")
    

if __name__ == "__main__":
    ctx = mp.get_context('spawn')
    
    
    import numpy as np
    import torch
    import os
    import time
    
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
    
    #from src.log_init                  import logger
    #from i24_logger.log_writer         import logger,catch_critical,log_warnings
    
    
    
    #from src.load.cpu_load             import DummyNoiseLoader #,MCLoader
    #from src.load.gpu_load             import MCLoader
    
    
    
    #%% Old run settings and setup
    if False: 
        run_config = "execute.config"       
        #os.environ["user_config_directory"] = "/home/derek/Documents/i24/i24_track/config/lambda_cerulean"
        #os.environ["TRACK_CONFIG_SECTION"] = "DEFAULT"
        in_dir = "/home/derek/Data/dataset_beta/sequence_1"
        
        
        # load parameters
        params = parse_cfg("TRACK_CONFIG_SECTION",
                            cfg_name=run_config, SCHEMA=False)
        
        # verify config notion of GPUs matches torch.cuda notion of available devices
        # get available devices
        
        # # initialize logger
        # try:
        #     log_params = {
        #                 "log_name":"Tracking Session: {}".format(np.random.randint(0,1000000)),
        #                 "processing_environment":os.environ["TRACK_CONFIG_SECTION"],
        #                 "logstash_address":(params.log_host_ip,params.log_host_port),
        #                 "connect_logstash": (True if "logstash" in params.log_mode else False),
        #                 "connect_syslog":(True if "syslog" in params.log_mode else False),
        #                 "connect_file": (True if "file" in params.log_mode else False),
        #                 "connect_console":(True if "sysout" in params.log_mode else False),
        #                 "console_log_level":params.log_level
        #                 }
            
        #     logger = connect_automatically(user_settings = log_params)
        #     logger.debug("Logger initialized with custom log settings",extra = log_params)
        # except:
        #     logger.debug("Logger initialized with default parameters")
        
        
        
        # TODO fix this once you redo configparse
        params.cuda_devices = [int(i) for i in range(int(params.cuda_devices))]
        assert max(params.cuda_devices) < torch.cuda.device_count()
        
        # intialize DeviceMap
        dmap = get_DeviceMap(params.device_map)
    
        from src.load.gpu_load             import MCLoader
        loader = MCLoader(in_dir, dmap.camera_mapping_file, ctx)
    
    
    
    #%% run settings    
    tm = Timer()
    tm.split("Init")
    
    run_config = "execute.config"       
    #mask = ["p46c01","p46c02", "p46c03", "p46c04", "p46c05","p46c06"]
    mask = None
    
    # load parameters
    params = parse_cfg_wrapper(run_config)
    
    in_dir = params.input_directory
    
    # # initialize logger
    # try:
    #     log_params = {
    #                 "log_name":"Tracking Session: {}".format(np.random.randint(0,1000000)),
    #                 "processing_environment":os.environ["TRACK_CONFIG_SECTION"],
    #                 "logstash_address":(params.log_host_ip,params.log_host_port),
    #                 "connect_logstash": (True if "logstash" in params.log_mode else False),
    #                 "connect_syslog":(True if "syslog" in params.log_mode else False),
    #                 "connect_file": (True if "file" in params.log_mode else False),
    #                 "connect_console":(True if "sysout" in params.log_mode else False),
    #                 "console_log_level":params.log_level
    #                 }
        
    #     logger = connect_automatically(user_settings = log_params)
    #     logger.debug("Logger initialized with custom log settings",extra = log_params)
    # except:
    #     logger.debug("Logger initialized with default parameters")
    
    # verify config notion of GPUs matches torch.cuda notion of available devices
    # get available devices
    
    # TODO fix this once you redo configparse
    params.cuda_devices = [int(i) for i in range(int(params.cuda_devices))]
    assert max(params.cuda_devices) < torch.cuda.device_count()
    
    # intialize DeviceMap
    dmap = get_DeviceMap(params.device_map)
    
    loader = MCLoader(in_dir, dmap.camera_mapping_file,dmap.cam_names, ctx)
    
    logger.debug("Initialized {} loader processes.".format(len(loader.device_loaders)))
    
    #%% more init stuff 
    
    # initialize Homography object
    hg = HomographyWrapper(hg1 = params.eb_homography_file,hg2 = params.wb_homography_file)
    
    # initialize pipelines
    pipelines = params.pipelines
    pipelines = [get_Pipeline(item, hg) for item in pipelines]
    
    associators = params.associators
    associators = [get_Associator(item) for item in associators]
    
    # initialize tracker
    tracker = get_Tracker(params.tracker)
    
    # add Associate function to each pipeline
    # for i in range(len(pipelines)):
    #     assoc = associators[i]
    #     pipelines[i].associate = associators[i]
    
    # initialize DetectorBank
    dbank = DeviceBank(params.cuda_devices, pipelines, dmap.gpu_cam_names, ctx)
    
    # initialize DBWriter object
    if params.write_db:
        dbw = WriteWrapper()
    
    # intialize empty TrackState Object
    tstate = TrackState()
    
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
    term_objects = 0
    
    # plot first frame
    if params.plot:
        plot_scene(tstate, frames, ts_trunc, dmap.gpu_cam_names,
             hg, colors,extents=dmap.cam_extents_dict, mask=mask,fr_num = frames_processed,detections = None)
    
    
    
    
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
            
            if True: # shortout actual processing
                
                # select pipeline for this frame
                tm.split("Predict")
                pidx = frames_processed % len(params.pipeline_pattern)
                pipeline_idx = params.pipeline_pattern[pidx]
        
                camera_idxs, device_idxs, obj_times = dmap(tstate, ts_trunc)
                obj_ids, priors, selected_obj_idxs = tracker.preprocess(
                    tstate, obj_times)
        
                # slice only objects we care to pass to DeviceBank on this set of frames
                # DEREK NOTE may run into trouble here since dmap and preprocess implicitly relies on the list ordering of tstate
                if len(obj_ids) > 0:
                    obj_ids     =     obj_ids[selected_obj_idxs]
                    priors      =      priors[selected_obj_idxs,:]
                    device_idxs = device_idxs[selected_obj_idxs]
                    camera_idxs = camera_idxs[selected_obj_idxs]
                
                # prep input stack by grouping priors by gpu
                tm.split("Map")
                cam_idx_names = None  # map idxs to names here
                prior_stack = dmap.route_objects(
                    obj_ids, priors, device_idxs, camera_idxs, run_device_ids=params.cuda_devices)
        
                # test on a single on-process pipeline
                # pipelines[pipeline_idx].set_device(0)
                # pipelines[pipeline_idx].set_cam_names(dmap.gpu_cam_names[0])
                # test = pipelines[pipeline_idx](frames[0],prior_stack[0])
        
        
                # TODO - full frame detections should probably get full set of objects?
                # TODO select correct pipeline based on pipeline pattern logic parameter
                tm.split("Detect {}".format(pipeline_idx),SYNC = True)
                detections, confs, classes, detection_cam_names, associations = dbank(
                    prior_stack, frames, pipeline_idx=pipeline_idx)
                
                # THIS MAY BE SLOW SINCE ITS DOUBLE INDEXING
                detection_times = torch.tensor(
                    [ts_trunc[dmap.cam_idxs[cam_name]] for cam_name in detection_cam_names])
                
                
                
                tm.split("Associate",SYNC = True)
                if pipeline_idx == 0:
                    detections_orig = detections.clone()
                    if True and len(detections) > 0:
                        # do nms across all device batches to remove dups
                        space_new = hg.state_to_space(detections)
                        keep = space_nms(space_new,confs)
                        detections = detections[keep,:]
                        classes = classes[keep]
                        confs = confs[keep]
                        detection_times = detection_times[keep]
                    
                    # overwrite associations here
                    associations = associators[0](obj_ids,priors,detections,hg)
        
                tm.split("Postprocess")
                terminated_objects = tracker.postprocess(
                    detections, detection_times, classes, confs, associations, tstate, hg = hg,measurement_idx =0)
                term_objects += len(terminated_objects)
    
                if params.write_db:
                    dbw.insert(terminated_objects,time_offset = start_ts)
                #print("Active Trajectories: {}  Terminated Trajectories: {}   Documents in database: {}".format(len(tstate),len(terminated_objects),len(dbw)))
    
            frames_processed += 1
    
            # optionally, plot outputs
            if params.plot:
                tm.split("Plot")
                detections = None
                priors = None
                plot_scene(tstate, frames, ts_trunc, dmap.gpu_cam_names,
                     hg, colors,extents=dmap.cam_extents_dict, mask=mask,fr_num = frames_processed,detections = detections,priors = priors)
    
            
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
                logger.info("Tracking Status Log",extra = metrics)
                logger.info("Time Utilization: {}".format(tm),extra = tm.bins())
                
                checkpoint(target_time,tstate)
       
        
        checkpoint(tstate, target_time)
        logger.info("Finished tracking over input time range. Shutting down.")
        
    except KeyboardInterrupt:
        logger.debug("Keyboard Interrupt recieved. Initializing soft shutdown")
        soft_shutdown(target_time, tstate,cleanup = [dbw,loader,dbank])
        
    
    	
