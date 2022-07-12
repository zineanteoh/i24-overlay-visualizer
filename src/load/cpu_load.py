import configparser
import torch

import os
import numpy as np
import time
import re

import cv2
from PIL import Image

from torchvision.transforms import functional as F

class DummyNoiseLoader():
    """
        Parses the config and returns random noise in the right shape
    """
    
    
    def __init__(self,mapping_file):
        
        # all we need to know which cameras go on which GPU
        
        cp = configparser.ConfigParser()
        cp.read(mapping_file)
        mapping = dict(cp["DEFAULT"])
       
        for key in mapping.keys():
            parsed_val = int(mapping[key])
            mapping[key] = parsed_val
    
        self.cam_devices = mapping

        self.dummy_time = 0         
            
    def get_frames(self,target_time):
        
        frames = [[] for i in range(torch.cuda.device_count())]
        for cam_id in self.cam_devices:
            dev_id = self.cam_devices[cam_id]
            frames[dev_id].append(torch.rand(3,1080//2,1920//2).to(torch.device("cuda:{}".format(dev_id))))
            
        # stack each list
        out = []
        for lis in frames:
            out.append(torch.stack(lis))
        ts = torch.tensor([self.dummy_time for i in range(len(self.cam_devices))])    
        
        return out,ts



class MCLoader():
    """
    Loads multiple video files in parallel with no timestamp parsing
    """
    
    def __init__(self,directory,mapping_file,ctx,resize = (1920,1080)):
        
        cp = configparser.ConfigParser()
        cp.read(mapping_file)
        mapping = dict(cp["DEFAULT"])
       
        for key in mapping.keys():
            parsed_val = int(mapping[key])
            mapping[key] = parsed_val
    
        self.cam_devices = mapping

        self.dummy_time = 0         

        cam_sequences = {}        
        for file in os.listdir(directory):
            try:
                sequence = os.path.join(directory,file)
                cam_name = re.search("p\dc\d",sequence).group(0)
                cam_sequences[cam_name] = sequence
            except:
                pass
        
        # device loader is a list of lists, with list i containing all loaders for device i (hopefully in order but not well enforced by dictionary so IDK)
        self.device_loaders = [[] for i in range(torch.cuda.device_count())]
        for key in cam_sequences.keys():
            dev_id = self.cam_devices[key]
            sequence = cam_sequences[key]
            loader = CV2BackendFrameGetter(sequence,dev_id,ctx,resize = resize)
            
            self.device_loaders[dev_id].append(loader)
        
        
            
    def get_frames(self,target_time):
        
        frames = [[] for i in range(torch.cuda.device_count())]
        for dev_idx,this_dev_loaders in enumerate(self.device_loaders):
            for loader in this_dev_loaders:
                frame = next(loader)
                frames[dev_idx].append(frame)
            
        # stack each list
        out = []
        for lis in frames:
            out.append(torch.stack(lis))
        ts = torch.tensor([self.dummy_time for i in range(len(self.cam_devices))])    
        
        self.dummy_time += 0.0333
        
        return out,ts


class CV2BackendFrameGetter():    
    def __init__(self,track_directory,device,ctx,buffer_size = 3,resize = (1920,1080)):
        
        """
        """
        
        # create shared queue
        self.queue = ctx.Queue()
        self.frame_idx = -1
        self.device = device

        try:
            files = []
            for item in [os.path.join(track_directory,im) for im in os.listdir(track_directory)]:
                files.append(item)
                files.sort()    
            
            self.length = len(files)
            self.files = files
            self.worker = ctx.Process(target=load_to_queue, args=(self.queue,files,device,buffer_size,resize))
            
        
        except: # file is a video         
            self.sequence = track_directory           
            self.frame_idx = -1
            
            test = cv2.VideoCapture(self.sequence)
            self.length = test.get(7)
            test.release()
            
            self.worker = ctx.Process(target=load_to_queue_video, args=(self.queue,self.sequence,device,buffer_size,resize))
        self.worker.start()
        time.sleep(0.01)
        
    def __len__(self):
        """
        Description
        -----------
        Returns number of frames in the track directory
        """
        return int(self.length)
    
    
    def __next__(self):
        """
        Description
        -----------
        Returns next frame and associated data unless at end of track, in which
        case returns -1 for frame num and None for frame

        Returns
        -------
        frame_num : int
            Frame index in track
        frame : tuple of (tensor,tensor,tensor)
            image, image dimensions and original image

        """
        
        if self.frame_idx < len(self) -1:
        
            frame = self.queue.get(timeout = 10)
            self.frame_idx = frame[0]
            return frame[1]
        
        else:
            self.worker.terminate()
            self.worker.join()
            return None

def load_to_queue(image_queue,files,det_step,init_frames,device,queue_size,resize):
    """
    Description
    -----------
    Whenever necessary, loads images, moves them to GPU, and adds them to a shared
    multiprocessing queue with the goal of the queue always having a certain size.
    Process is to be called as a worker by FrameLoader object
    
    Parameters
    ----------
    image_queue : multiprocessing Queue
        shared queue in which preprocessed images are put.
    files : list of str
        each str is path to one file in track directory
    det_step : int
        specifies number of frames between dense detections 
    init_frames : int
        specifies number of dense detections before localization begins
    device : torch.device
        Specifies whether images should be put on CPU or GPU.
    queue_size : int, optional
        Goal size of queue, whenever actual size is less additional images will
        be processed and added. The default is 5.
    """
    
    frame_idx = 0    
    while frame_idx < len(files):
        
        if image_queue.qsize() < queue_size:
            
            # load next image
            with Image.open(files[frame_idx]) as im:
             
              # if frame_idx % det_step.value < init_frames:   
              #     # convert to CV2 style image
              #     open_cv_image = np.array(im) 
              #     im = open_cv_image.copy() 
              #     original_im = im[:,:,[2,1,0]].copy()
              #     # new stuff
              #     dim = (im.shape[1], im.shape[0])
              #     im = cv2.resize(im, (1920,1080))
              #     im = im.transpose((2,0,1)).copy()
              #     im = torch.from_numpy(im).float().div(255.0).unsqueeze(0)
              #     dim = torch.FloatTensor(dim).repeat(1,2)
              #     dim = dim.to(device,non_blocking = True)
              # else:
                  # keep as tensor
              original_im = np.array(im)[:,:,[2,1,0]].copy()
              im = F.resize(im,resize)
              im = F.to_tensor(im)
              im = F.normalize(im,mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                 
              # store preprocessed image, dimensions and original image
              im = im.to(device)
              frame = (frame_idx,im)
             
              # append to queue
              image_queue.put(frame)
             
            frame_idx += 1
    
    # neverending loop, because if the process ends, the tensors originally
    # initialized in this function will be deleted, causing issues. Thus, this 
    # function runs until a call to self.next() returns -1, indicating end of track 
    # has been reached
    while True:  
           time.sleep(5)
        
def load_to_queue_video(image_queue,sequence,device,queue_size,resize):
    
    # checksum_path="/home/worklab/Documents/derek/I24-video-processing/I24-video-ingest/resources/timestamp_pixel_checksum_6.pkl"
    # geom_path="/home/worklab/Documents/derek/I24-video-processing/I24-video-ingest/resources/timestamp_geometry_4K.pkl"
    # checksums = tsu.get_precomputed_checksums(checksum_path)
    # geom = tsu.get_timestamp_geometry(geom_path)
    
    cap = cv2.VideoCapture(sequence)
    
    frame_idx = 0    
    while frame_idx < 30*5*60:
        
        if image_queue.qsize() < queue_size:
            
            # load next image from videocapture object
            ret,original_im = cap.read()
            if ret == False:
                frame = (-1,None)
                image_queue.put(frame)       
                break
            else:
                original_im = cv2.resize(original_im,resize)
                im = F.to_tensor(original_im)
                im = F.normalize(im,mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
                # store preprocessed image, dimensions and original image
                im = im.to(device)
                frame = (frame_idx,im)
             
                # append to queue
                image_queue.put(frame)       
                frame_idx += 1
    
    # neverending loop, because if the process ends, the tensors originally
    # initialized in this function will be deleted, causing issues. Thus, this 
    # function runs until a call to self.next() returns -1, indicating end of track 
    # has been reached
    while True:  
           time.sleep(5)
    
    