# i24_overlay_visualizer

Overlay bounding boxes visualizer for i24 motion. 

## Notes:

1. This current repository accesses the os.environ["USER_CONFIG_DIRECTORY"] to find the path to config folder. 
    Running this i24_video_viz on Derek's computer will access the config file in i24_track
    i.e. os.environ["USER_CONFIG_DIRECTORY"]="/home/derek/Documents/i24/i24_video_viz/config/lambda_cerulean_2"
    
    TODO: specify the config directory to point to ./config/lambda_cerulean_2
    
2. `misc.py` contains the function `plot_scene()`, which calls 
        - `transform_raw_docs(...)`         if the transformed collection to visualize contains RAW trajectories
            - transformed collection already contains 'dimensions' field
        - `transform_reconciled_docs(...)`  if the transformed collection to visualize contains RECONCILED trajectories