from i24_database_api.db_writer import DBWriter
from i24_configparse import parse_cfg
from i24_logger.log_writer import catch_critical,logger

import os
import numpy as np


class WriteWrapper():
    
    @catch_critical()
    def __init__(self,server_id = -1):
    
        
    
        self.SESSION_CONFIG_ID = os.environ["TRACK_CONFIG_SECTION"]
        self.PID = os.getpid()
        self.COMPUTE_NODE_ID = server_id 
        
        self = parse_cfg("TRACK_CONFIG_SECTION",obj=self)    
    
        
        self.dbw = DBWriter(
                       host               = self.host, 
                       port               = self.port, 
                       username           = self.username, 
                       password           = self.password,
                       database_name      = self.db_name, 
                       schema_file        = self.schema_file,
                       collection_name    = self.raw_collection,
                       server_id          = self.COMPUTE_NODE_ID, 
                       session_config_id  = self.SESSION_CONFIG_ID,
                       process_id         = self.PID,
                       process_name       = "track"
                       )
    
    
    
        col = self.dbw.collection
        
        
        logger.debug("Initialized log writer to Collection {} ({} existing records)".format(self.raw_collection,len(self)))
        
        self.prev_len = len(self) -1
        self.prev_doc = None
    
    @catch_critical()
    def __len__(self):
        return self.dbw.collection.count_documents({})
    
    
    @catch_critical()
    def insert(self,trajectories,time_offset = 0):
        """
        Converts trajectories as dequeued from TrackState into document form and inserts with dbw
        trajectories - output from TrackState.remove()
        """
        
        if len(trajectories) == 0:
            return
        
        # cur_len = len(self)
        # if cur_len == self.prev_len:
        #     logger.warning("\n Document {} was not correctly inserted into database collection {}".format(self.prev_doc,self.raw_collection))
        # self.prev_len = cur_len
        
        for id in trajectories.keys():
            trajectory = trajectories[id]
            history = trajectory[0]
            cls_data = trajectory[1]
            
            cls = int(np.argmax(cls_data))
            timestamps = [item[0] + time_offset for item in history]
            x = [item[1][0].item() for item in history]
            y = [item[1][1].item() for item in history]
            l = [item[1][2].item() for item in history]
            w = [item[1][3].item() for item in history]
            h = [item[1][4].item() for item in history]
    
            # convert to document form
            doc = {}
            doc["configuration_id"]        = self.SESSION_CONFIG_ID
            doc["local_fragment_id"]       = id
            doc["compute_node_id"]         = self.COMPUTE_NODE_ID
            doc["coarse_vehicle_class"]    = cls
            doc["fine_vehicle_class"]      = -1
            doc["timestamp"]               = timestamps
            doc["raw timestamp"]           = timestamps
            doc["first_timestamp"]         = timestamps[0]
            doc["last_timestamp"]          = timestamps[-1]
            doc["road_segment_ids"]        = [-1]
            doc["x_position"]              = x
            doc["y_position"]              = y
            doc["starting_x"]              = x[0]
            doc["ending_x"]                = x[-1]
            doc["camera_snapshots"]        = "None"
            doc["flags"]                   = ["test flag 1","test flag 2"]
            doc["length"]                  = l
            doc["width"]                   = w
            doc["height"]                  = h
            doc["direction"]               = -1 if y[0] > 60 else 1
            
            
            
            
            # insert
            if len(x) > self.min_document_length:
                self.dbw.write_one_trajectory(**doc) 
                
    
            
if __name__ == "__main__":
    test = WriteWrapper()