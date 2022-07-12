from .kf import Torch_KF 
from i24_configparse import parse_cfg
import _pickle as pickle
import torch
import numpy as np




class TrackState():
    """
    TrackState object maintains the state of the tracking problem. This means storing:
        a. the current state (in a filtering state formulation)
        b. historical states of all active objects
    TrackState objects deal with objects natively in a space-based representation, so 
    object positions are expected in and are returned in state-space coordinates
    
    The following functions are implemented:
        - __call__() wraps dict() or tensor() methods
        - predict() - wrapper around KF predict function
        - update() - wrapper areound KF update that also updates the stored history
        - remove() - removes a set of objects and returns that set
        - add() - adds a set of objects
    """
    
    
    def __init__(self):
        self.fsld = {}                   # fsld[id] stores frames since last detected for object id
        self.all_classes = {}            # dict of lists of integer classes for each object
        self.all_confs = {}              # dict of lists of float confidences for each object
        self._history = {}                # dict of lists, each list item is (time,kf_state) 
        self._next_obj_id = 0             # next id for a new object (incremented during tracking)

        
        # load config params (self.device_id, self.kf_param_path,self.n_classes)
        self = parse_cfg("TRACK_CONFIG_SECTION",obj = self)
        
        self.device = torch.device("cpu") if self.device_id == -1 else torch.device("cuda:{}".format(self.device_id))
        
        if self.kf_param_path is not None:
            with open(self.kf_param_path ,"rb") as f:
                kf_params = pickle.load(f)
        
        # initialize Kalman filter
        self.kf = Torch_KF(self.device,INIT = kf_params)
        
    def __len__(self):      
        try:
            return self.kf.X.shape[0]
        except AttributeError: # no objects in self.kf.X
            return 0
     
    def __call__(self,target_time = None,with_direction = True,mode = "tensor"):
        """
        returns current state of tracked objects
        :param target_time (float) - time at which the position of objects should be returned
                                  if None, each state is returned at latest time
        :param with_direction (bool) - if True, direction included in state
        :param mode (string) - modulates return type, must be either tensor or dict
        :returns - dict with state keyed by ID, or tensor of IDs and tensor of states
        """
        
        if target_time is not None:
            ids, states = self.kf.view(dt = self.kf.get_dt(target_time),with_direction = with_direction)
        else:
            ids, states = self.kf.view(dt = None,with_direction = with_direction)

        if mode == "tensor":
            return torch.tensor(ids), states
    
        else:
            state_dict = dict([(ids[i],states[i]) for i in range(len(ids))])
            return state_dict
        
    def add(self,detections,directions,detection_times,labels,scores,init_speed = False):
        
        new_ids = []
        
        for i in range(len(detections)):                
            new_ids.append(self._next_obj_id)

            # add internal attributes for new object            
            self.fsld[self._next_obj_id] = 0
            self.all_classes[self._next_obj_id] = np.zeros(self.n_classes)
            self.all_confs[self._next_obj_id] = []
            self._history[self._next_obj_id] = []
            
            # get class and conf for object
            cls = int(labels[i])
            self.all_classes[self._next_obj_id][cls] += 1
            self.all_confs[self._next_obj_id].append(scores[i])         
            
            self._next_obj_id += 1
            
        if len(detections) > 0:   
            self.kf.add(detections,new_ids,directions,detection_times,init_speed = init_speed)
           
        # update history
        for id in new_ids:
            self._update_history(id)
        
      
    def _update_history(self,id):
        time = self.kf.T[self.kf.obj_idxs[id]].item()
        self._history[id].append((time,self.kf.X[self.kf.obj_idxs[id]]))
        
    def remove(self,ids):
        removals = {}
        for id in ids:
            # store history
            datum = []
            datum.append(self._history.pop(id))
            datum.append(self.all_classes.pop(id))
            removals[id] = datum
        
            del self.all_confs[id]
            del self.fsld[id]
            
        self.kf.remove(ids)
        return removals
    
    def get_classes(self):
        classes = {}
        for id in self.all_classes.keys():
            classes[id] = np.argmax(self.all_classes[id]) 
        return classes
    
    def get_lifespans(self):
        lifespans = {}
        for id in self._history.keys():
            lifespans[id] = len(self._history[id])
        return lifespans
    
    def get_dt(self,target_times,idxs = None):
        return self.kf.get_dt(target_times,idxs = idxs)
    
    def predict(self,dt = None):
        self.kf.predict(dt = dt)
    
    def update(self,detections,obj_ids,classes,confs,measurement_idx = 1,high_confidence_threshold = 0):
        if len(obj_ids) > 0:
            # update kf states
            self.kf.update(detections,obj_ids,measurement_idx = measurement_idx)
        
        # increment all fslds - any obj with an update will have fsld overwritten next
        for id in self.fsld.keys():
            self.fsld[id] += 1 
        
        # update fsld, class, and conf
        for i,id in enumerate(obj_ids):
            self.all_classes[id][int(classes[i])] += 1
            self.all_confs[id].append(confs[i])
            
            if confs[i] > high_confidence_threshold:
                self.fsld[id] = 0
                
        # update stored history
        for id in obj_ids:
            self._update_history(id)
            
        
        
    
        