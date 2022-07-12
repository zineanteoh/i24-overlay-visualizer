from gpu_load_multi       import MCLoader, ManagerClock

loader = MCLoader(in_dir, dmap.camera_mapping_file,dmap.cam_names, ctx,start_time = target_time)
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
