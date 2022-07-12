import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from ..util.bbox import im_nms,space_nms
from i24_configparse import parse_cfg


def get_Tracker(name):
    """
    getter function that takes a string (class name) input and returns an instance
    of the named class
    """
    if name == "BaseTracker":
        tracker = BaseTracker()
    elif name == "SmartTracker":
        tracker = SmartTracker()  
    else:
        raise NotImplementedError("No BaseTracker child class named {}".format(name))
    
    return tracker

def get_Associator(name,device_id = -1):
    """
    getter function that takes a string (class name) input and returns an instance
    of the named class
    """
    if name == "Associator":
        assoc = Associator()
    elif name == "HungarianIOUAssociator":
        assoc = HungarianIOUAssociator()
    else:
        raise NotImplementedError("No Associator child class named {}".format(name))
    
    return assoc



class Associator():

    def __init__(self):
        self = parse_cfg("TRACK_CONFIG_SECTION",obj = self)
        
        #self.device = torch.cuda.device("cuda:{}".format(self.device_id) if self.device_id != -1 else "cpu")
        
    def __call__(self,obj_ids,priors,detections,hg = None):
        """
        Applies an object association strategy to pair detections to priors
        :param obj_ids- tensor of length n_objs with unique integer ID for each
        :param priors - tensor of size [n_objs,state_size] with object postions for each
        :param detections - tensor of size [n_detections,state_size] with detection positions for each
        :param hg - HomographyWrapper object
        :return associations - tensor of size n_detections with ID for each detection or -1 if no ID is associated
        """
        
        raise NotImplementedError
        
        
class HungarianIOUAssociator(Associator):
    def __init__(self):
        self = parse_cfg("DEFAULT",obj = self)
        
        #self.device = torch.cuda.device("cuda:{}".format(self.device_id) if self.device_id != -1 else "cpu")
        #print("Matching min IOU: {}".format(self.min_match_iou))
    
    def __call__(self,obj_ids,priors,detections,hg):
       """
       Applies association logic by intersection-over-union metric and Hungarian 
       algorithm for bipartite matching
       
       :param obj_ids- tensor of length n_objs with unique integer ID for each
       :param priors - tensor of size [n_objs,state_size] with object postions for each
       :param detections - tensor of size [n_detections,state_size] with detection positions for each
       :param hg - HomographyWrapper object
       :return associations - tensor of size [n_detections] with ID for each detection or -1 if no ID is associated
       """
       
       # aliases
       first = priors
       second = detections
       
       if len(second) == 0:
            return torch.empty(0)
       
       if len(first) == 0:   
           return torch.zeros(len(second))-1
       
       # print("Priors:")
       # print(first)
       # print("Detections:")
       # print(second)



       # first and second are in state form - convert to space form
       first = hg.state_to_space(first.clone())
       boxes_new = torch.zeros([first.shape[0],4],device = first.device)
       boxes_new[:,0] = torch.min(first[:,0:4,0],dim = 1)[0]
       boxes_new[:,2] = torch.max(first[:,0:4,0],dim = 1)[0]
       boxes_new[:,1] = torch.min(first[:,0:4,1],dim = 1)[0]
       boxes_new[:,3] = torch.max(first[:,0:4,1],dim = 1)[0]
       first = boxes_new
       
       second = hg.state_to_space(second.clone())
       boxes_new = torch.zeros([second.shape[0],4],device = second.device)
       boxes_new[:,0] = torch.min(second[:,0:4,0],dim = 1)[0]
       boxes_new[:,2] = torch.max(second[:,0:4,0],dim = 1)[0]
       boxes_new[:,1] = torch.min(second[:,0:4,1],dim = 1)[0]
       boxes_new[:,3] = torch.max(second[:,0:4,1],dim = 1)[0]
       second = boxes_new
       
       f = first.shape[0]
       s = second.shape[0]
       
       #get weight matrix
       second = second.unsqueeze(0).repeat(f,1,1).double()
       first = first.unsqueeze(1).repeat(1,s,1).double()
       dist = self.md_iou(first,second)

       try:
           a, b = linear_sum_assignment(dist.data.numpy(),maximize = True) 
       except ValueError:
            return torch.zeros(s)-1
            print("DEREK USE LOGGER WARNING HERE")
        
       
       # convert into expected form
       matchings = np.zeros(s)-1
       for idx in range(0,len(b)):
            matchings[b[idx]] = a[idx]
       matchings = np.ndarray.astype(matchings,int)
        
       # remove any matches too far away
       # TODO - Vectorize this
       for i in range(len(matchings)):
           if matchings[i] != -1 and  dist[matchings[i],i] < (self.min_match_iou):
               matchings[i] = -1    
    
        # matchings currently contains object indexes - swap to obj_ids
       for i in range(len(matchings)):
           if matchings[i] != -1:
               matchings[i] = obj_ids[matchings[i]]
               
       # print("Dist:")
       # print(dist)
       # print("Matches:")
       # print(matchings)
        
       return torch.from_numpy(matchings)
    
            
   
    def md_iou(self,a,b):
        """
        a,b - [batch_size ,num_anchors, 4]
        """
        
        area_a = (a[:,:,2]-a[:,:,0]) * (a[:,:,3]-a[:,:,1])
        area_b = (b[:,:,2]-b[:,:,0]) * (b[:,:,3]-b[:,:,1])
        
        minx = torch.max(a[:,:,0], b[:,:,0])
        maxx = torch.min(a[:,:,2], b[:,:,2])
        miny = torch.max(a[:,:,1], b[:,:,1])
        maxy = torch.min(a[:,:,3], b[:,:,3])
        zeros = torch.zeros(minx.shape,dtype=float,device = a.device)
        
        intersection = torch.max(zeros, maxx-minx) * torch.max(zeros,maxy-miny)
        union = area_a + area_b - intersection
        iou = torch.div(intersection,union)
        
        #print("MD iou: {}".format(iou.max(dim = 1)[0].mean()))
        return iou






class BaseTracker():
    """
    Basic wrapper around TrackState that adds some basic functionality for object 
    management and stale object removal. The thinking was that this may be nonstandard
    so should be abstracted away from the TrackState object, but the two classes
    may be unified in a future version.
    """
    
    def __init__(self):
        
        # parse config file
        self = parse_cfg("TRACK_CONFIG_SECTION",obj = self)    
        

    def preprocess(self,tstate,obj_times):
        """
        Receives a TrackState object as input, as well as the times for each object
        Applies kf.predict to objects
        Theoretically, additional logic could happen here i.e. a subset of objects
        could be selected for update based on covarinace, etc.
        
        :param tstate - TrackState object
        :param obj_times - tensor of times of same length as number of active objects
        
        :return obj_ids - tensor of int IDs for object priors
        :return priors - tensor of all TrackState object priors
        :return selected_idxs - tensor of idxs on which to perform measurement update (naively, all of them)
        """
        
        dts = tstate.get_dt(obj_times)
        tstate.predict(dt = dts)   
        
        obj_ids,priors = tstate()
        
        selected_idxs = self.select_idxs(tstate,obj_times)
        
        return obj_ids,priors,selected_idxs
        
    def select_idxs(self,tstate,obj_times):
        return torch.tensor([i for i in range(len(tstate))])
        

    def postprocess(self,detections,detection_times,classes,confs,assigned_ids,tstate,hg = None,measurement_idx = 0):
        """
        Updates KF representation of objects where assigned_id is not -1 (unassigned)
        Adds other objects as new detections
        For all TrackState objects, checks confidences and fslds and removes inactive objects
        :param detections -tensor of size [n_detections,state_size]
        :param detection_times - tensor of size [n_detections] with frame time
        :param classes - tensor of size [n_detections] with integer class prediction for each
        :param confs - tensor of size [n_detections] of confidences in range[0,1]
        :param assigned_ids - tensor of size [n_detections] of IDs, or -1 if no id assigned
        :param tstate - TrackState object
        :param meas_idx - int specifying which measurement type was used
        
        :return - stale_objects - dictionary of object histories indexed by object ID
        """
        
        # get IDs and times for update
        if len(assigned_ids) > 0:
            update_idxs = torch.nonzero(assigned_ids + 1).squeeze(1) 
            
            if len(update_idxs) > 0:
                update_ids = assigned_ids[update_idxs].tolist()
                update_times = detection_times[update_idxs]
                
                tstate_ids = tstate()[0]
                for id in update_ids:
                    assert (id in tstate_ids), "{}".format(id)
                    
                

                # TODO this may give an issue when some but not all objects need to be rolled forward
                # roll existing objects forward to the detection times
                dts = tstate.get_dt(update_times,idxs = update_ids)
                tstate.predict(dt = dts)
            
            
                # update assigned detections
                update_detections = detections[update_idxs,:]
                update_classes = classes[update_idxs]
                update_confs = confs[update_idxs]
                tstate.update(update_detections[:,:5],update_ids,update_classes,update_confs, measurement_idx = measurement_idx)
            
            # collect unassigned detections
            new_idxs = [i for i in range(len(assigned_ids))]
            for i in update_idxs:
                new_idxs.remove(i)
              
            
            # add new detections as new objects
            if len(new_idxs) > 0:
                new_idxs = torch.tensor(new_idxs)
                new_detections = detections[new_idxs,:]
                new_classes = classes[new_idxs]
                new_confs = confs[new_idxs]
                new_times = detection_times[new_idxs]
                
                
                # # do nms across all device batches to remove dups
                # if hg is not None:
                #     space_new = hg.state_to_space(new_detections)
                #     keep = space_nms(space_new,new_confs)
                #     new_detections = detections[keep,:]
                #     new_classes = classes[keep]
                #     new_confs = confs[keep]
                #     new_times = detection_times[keep]
                
                # create direction tensor based on location
                directions = torch.where(new_detections[:,1] > 60, torch.zeros(new_confs.shape)-1,torch.ones(new_confs.shape))
                
                tstate.add(new_detections,directions,new_times,new_classes,new_confs,init_speed = True)
            
            
            # do nms on all objects to remove overlaps, where score = # of frames since initialized
            if  hg is not None:
                ids,states = tstate()
                space = hg.state_to_space(states)
                lifespans = tstate.get_lifespans()
                scores = torch.tensor([lifespans[id.item()] for id in ids])
                keep = space_nms(space,scores.float(),threshold = 0.1)
                keep_ids = ids[keep]
                
                removals = ids.tolist()
                for id in keep_ids:
                    removals.remove(id)
                    
                tstate.remove(removals)
                
            # remove anomalies
            ids,states = tstate()
            removals = []
            for i in range(len(ids)):
                if states[i][2] > 80 or states[i][2] < 10 or states[i][3] > 15 or states[i][3] < 3 or states[i][4] > 16 or states[i][4] < 3:
                    removals.append(ids[i].item())
            tstate.remove(removals)
            
          
        # if no detections, increment fsld in all tracked objects
        else:
            tstate.update(None,[],None,None)
        
        # remove objects
        stale_objects = self.remove(tstate)
        
        return stale_objects


    # TODO - remove this hard-coded removal rule
    def remove(self,tstate):
        """
        
        """
        # remove stale objects
        removals = []
        ids,states = tstate()
        for i in range(len(ids)):
            id = ids[i].item()
            if tstate.fsld[id] > self.fsld_max or states[i][0] < -200 or states[i][0] > 1800:
                removals.append(id)
                
        stale_objects = tstate.remove(removals)
        return stale_objects
    
 
class SmartTracker(BaseTracker):
    
    def postprocess(self,
                    detections,
                    detection_times,
                    classes,
                    confs,
                    assigned_ids,
                    tstate,
                    hg = None,
                    measurement_idx = 0):
        """
        Updates KF representation of objects where assigned_id is not -1 (unassigned)
        Adds other objects as new detections
        For all TrackState objects, checks confidences and fslds and removes inactive objects
        :param detections -tensor of size [n_detections,state_size]
        :param detection_times - tensor of size [n_detections] with frame time
        :param classes - tensor of size [n_detections] with integer class prediction for each
        :param confs - tensor of size [n_detections] of confidences in range[0,1]
        :param assigned_ids - tensor of size [n_detections] of IDs, or -1 if no id assigned
        :param tstate - TrackState object
        :param meas_idx - int specifying which measurement type was used
        
        :return - stale_objects - dictionary of object histories indexed by object ID
        """

        # get IDs and times for update
        if len(assigned_ids) > 0:
            update_idxs = torch.nonzero(assigned_ids + 1).squeeze(1) 
            
            if len(update_idxs) > 0:
                update_ids = assigned_ids[update_idxs].tolist()
                update_times = detection_times[update_idxs]
                
                tstate_ids = tstate()[0]
                for id in update_ids:
                    assert (id in tstate_ids), "{}".format(id)
                    
                

                # TODO this may give an issue when some but not all objects need to be rolled forward
                # roll existing objects forward to the detection times
                dts = tstate.get_dt(update_times,idxs = update_ids)
                tstate.predict(dt = dts)
            
            
                # update assigned detections
                update_detections = detections[update_idxs,:]
                update_classes = classes[update_idxs]
                update_confs = confs[update_idxs]
                tstate.update(update_detections[:,:5],update_ids,update_classes,update_confs, measurement_idx = measurement_idx,high_confidence_threshold = self.sigma_high)
            
            # collect unassigned detections
            new_idxs = [i for i in range(len(assigned_ids))]
            for i in update_idxs:
                new_idxs.remove(i)
              
            
            # add new detections as new objects
            if len(new_idxs) > 0:
                new_idxs = torch.tensor(new_idxs)
                new_detections = detections[new_idxs,:]
                new_classes = classes[new_idxs]
                new_confs = confs[new_idxs]
                new_times = detection_times[new_idxs]
                
                
                # create direction tensor based on location
                directions = torch.where(new_detections[:,1] > 60, torch.zeros(new_confs.shape)-1,torch.ones(new_confs.shape))
                
                tstate.add(new_detections,directions,new_times,new_classes,new_confs,init_speed = True)
                
        # if no detections, increment fsld in all tracked objects
        else:
            tstate.update(None,[],None,None)
            
        stale_objects = self.remove(tstate,hg)
        return stale_objects
            
    def remove(self,tstate,hg = None):
        
            
            # 1. do nms on all objects to remove overlaps, where score = # of frames since initialized
            if  hg is not None:
                ids,states = tstate()
                space = hg.state_to_space(states)
                lifespans = tstate.get_lifespans()
                scores = torch.tensor([lifespans[id.item()] for id in ids])
                keep = space_nms(space,scores.float(),threshold = 0.01)
                keep_ids = ids[keep]
                
                removals = ids.tolist()
                for id in keep_ids:
                    removals.remove(id)
                
                tstate.remove(removals)

                
            # 2. Remove anomalies
            ids,states = tstate()
            removals = []
            for i in range(len(ids)):
                for j in range(5):
                    if states[i,j] < self.state_bounds[2*j] or states[i,j] > self.state_bounds[2*j+1]:
                        removals.append(ids[i].item())
                        break
            tstate.remove(removals)

            
            # 3. Remove objects that don't have enough high confidence detections
            ids,states = tstate()
            removals = []
            for id in ids:
                id = id.item()
                if len(tstate.all_confs[id]) == self.n_init:
                    for conf in tstate.all_confs[id]:
                        if conf < self.sigma_high:
                            removals.append(id)
                            break
            tstate.remove(removals)


            # 4. Pop objects that are out of FOV
            removals = []
            ids,states = tstate()
            for i in range(len(ids)):
                id = ids[i].item()
                if tstate.fsld[id] > self.fsld_max or states[i][0] < self.fov[0] or states[i][0] > self.fov[1]:
                    removals.append(id)
                    
            stale_objects = tstate.remove(removals)

            return stale_objects