import torch
import PyNvCodec as nvc
import PytorchNvCodec as pnvc


import configparser
import torch

import os
import numpy as np
import time
import re

import cv2
from PIL import Image

from torchvision.transforms import functional as F

from i24_logger.log_writer import logger,catch_critical

class ManagerClock:
    def __init__(self,start_ts,desired_processing_speed,framerate):
        """
        start_ts - unix timestamp
        desired_processing_speed - int (if 1, realtime processing expected, if 0 no processing speed constraint,etc)
        framerate - int nominal capture speed of camera
        """

        self.dps = desired_processing_speed
        self.framerate = framerate
        
        self.start_ts = start_ts
        self.start_t = time.time()
        
    def tick(self,ts):
        """
        Returns the later of:
            1. max ts from previous camera timestamps + framerate
            2. the timestamp at which processing should be to maintain desired processing speed
        
        ts - list of lists of timestamps returned by loader
        """
        # flat_ts = []
        # for item in ts:
        #     flat_ts += item
        # max_ts = max(flat_ts)
        
        max_ts = max(ts)
        
        target_ts_1 = max_ts + 1.0/self.framerate
        
        
        elapsed_proc_time = time.time() - self.start_t
        
        target_ts_2 = self.start_ts + elapsed_proc_time*self.dps
        
        
        target_ts = max(target_ts_1,target_ts_2)
        return target_ts

class MCLoader():
    """
    Loads multiple video files in parallel with PTS timestamp decoding and 
    directory - overall file buffer
    """
    
    @catch_critical()        
    def __init__(self,directory,mapping_file,cam_names,ctx,resize = (1920,1080), start_time = None):
        
        cp = configparser.ConfigParser()
        cp.read(mapping_file)
        mapping = dict(cp["DEFAULT"])
        
        # TODO - use regex from config file rather than hard-coding it
               
        for key in mapping.keys():
            parsed_val = int(mapping[key])
            mapping[key] = parsed_val
    
        self.cam_devices = mapping


        # instead of getting individual files, sequence is a directorie (1 per camera)
        cam_sequences = {}        
        for file in os.listdir(directory):
            sequence = os.path.join(directory,file)
            cam_name = re.search("P\d\dC\d\d",sequence).group(0)
            cam_sequences[cam_name] = sequence
        
        # device loader is a list of lists, with list i containing all loaders for device i (hopefully in order but not well enforced by dictionary so IDK)
        self.device_loaders = [[] for i in range(torch.cuda.device_count())]
        for key in cam_names:
            dev_id = self.cam_devices[key.lower()]
            
            try:
                sequence = cam_sequences[key]
            except:
                sequence = cam_sequences[key.upper()]
            
            loader = GPUBackendFrameGetter(sequence,dev_id,ctx,resize = resize,start_time = start_time)
            
            self.device_loaders[dev_id].append(loader)
        
        
    @catch_critical()     
    def get_frames(self,target_time = None, tolerance = 1/60):
        
        # each camera gets advanced until it is
        # within tolerance of the target time
        
        if target_time is None:
            # accumulators
            frames = [[] for i in range(torch.cuda.device_count())]
            timestamps = []
            for dev_idx,this_dev_loaders in enumerate(self.device_loaders):
                for loader in this_dev_loaders:
                    frame,ts = next(loader)
                    frames[dev_idx].append(frame)
                    timestamps.append(ts)
        else:
            # accumulators
            frames = [[] for i in range(torch.cuda.device_count())]
            timestamps = []
            
            # advance each camera loader
            for dev_idx,this_dev_loaders in enumerate(self.device_loaders):
                for loader in this_dev_loaders:
                    
                    # require an advancement of at least one frame
                    frame,ts = next(loader)
                    
                    # skip frames as necessary to sync times
                    while ts + tolerance < target_time:
                        frame,ts = next(loader)
                    
                    frames[dev_idx].append(frame)
                    timestamps.append(ts)
                
        # stack each accumulator list
        out = []
        for lis in frames:
            if len(lis) == 0: # occurs when no frames are mapped to a GPU
                out.append(torch.empty(0))
            else:
                out.append(torch.stack(lis))

        #timestamps = torch.tensor([torch.tensor(item) for item in timestamps],dtype = torch.double)
        
        return out,timestamps
    
class GPUBackendFrameGetter:
    def __init__(self,directory,device,ctx,buffer_size = 5,resize = (1920,1080),start_time = None):
        
        # create shared queue
        self.queue = ctx.Queue()
        self.frame_idx = -1
        self.device = device  
        
        self.directory = directory
        # instead of a single file, pass a directory, and a start time
        self.worker = ctx.Process(target=load_queue_continuous_vpf, args=(self.queue,directory,device,buffer_size,resize,start_time))
        self.worker.start()   
            

    def __len__(self):
        """
        Description
        -----------
        Returns number of frames in the track directory
        """
        
        return 1000000
    
    
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
        
        
        frame = self.queue.get(timeout = 10)
        ts = frame[1] / 10e8
        im = frame[0]
        
        return im,ts
        
        # if False: #TODO - implement shutdown
        #     self.worker.terminate()
        #     self.worker.join()
        #     return None
        
def load_queue_continuous_vpf(q,directory,device,buffer_size,resize,start_time):
    
    logger.set_name("Hardware Decode Handler {}".format(device))
    
    resize = (resize[1],resize[0])
    gpuID = device
    device = torch.cuda.device("cuda:{}".format(gpuID))
    
    
    
    # GET FIRST FILE
    # sort directory files (by timestamp)
    files = os.listdir(directory)
    
    # filter out non-video_files and sort video files
    files = list(filter(  (lambda f: True if ".mkv" in f else False) ,   files))
    files.sort()
    
    # select next file that comes sequentially after last_file
    for fidx,file in enumerate(files):
        try:
            ftime = float(         file.split("_")[-1].split(".mkv")[0])
            nftime= float(files[fidx+1].split("_")[-1].split(".mkv")[0])
            if nftime >= start_time:
                break
        except:
            break # no next file so this file should be the one
    
    logger.debug("Loading frames from {}".format(file))
    last_file = file
    while True:
        
        file = os.path.join(directory,file)
        
        # initialize Decoder object
        nvDec = nvc.PyNvDecoder(file, gpuID)
        target_h, target_w = nvDec.Height(), nvDec.Width()
    
        to_rgb = nvc.PySurfaceConverter(nvDec.Width(), nvDec.Height(), nvc.PixelFormat.NV12, nvc.PixelFormat.RGB, gpuID)
        to_planar = nvc.PySurfaceConverter(nvDec.Width(), nvDec.Height(), nvc.PixelFormat.RGB, nvc.PixelFormat.RGB_PLANAR, gpuID)
    
        cspace, crange = nvDec.ColorSpace(), nvDec.ColorRange()
        if nvc.ColorSpace.UNSPEC == cspace:
            cspace = nvc.ColorSpace.BT_601
        if nvc.ColorRange.UDEF == crange:
            crange = nvc.ColorRange.MPEG
        cc_ctx = nvc.ColorspaceConversionContext(cspace, crange)
        
        
        # get frames from one file
        while True:
            if q.qsize() < buffer_size:
                pkt = nvc.PacketData()
                
                # advance frames until reaching start_time
                # Double check this math, pkt.pts is in nanoseconds I believe
                if start_time is not None and start_time > pkt.pts:
                    continue
                    
                
                # Obtain NV12 decoded surface from decoder;
                raw_surface = nvDec.DecodeSingleSurface(pkt)
                if raw_surface.Empty():
                    break
    
                # Convert to RGB interleaved;
                rgb_byte = to_rgb.Execute(raw_surface, cc_ctx)
            
                # Convert to RGB planar because that's what to_tensor + normalize are doing;
                rgb_planar = to_planar.Execute(rgb_byte, cc_ctx)
            
                # likewise, end of video file
                if rgb_planar.Empty():
                    break
                
                # Create torch tensor from it and reshape because
                # pnvc.makefromDevicePtrUint8 creates just a chunk of CUDA memory
                # and then copies data from plane pointer to allocated chunk;
                surfPlane = rgb_planar.PlanePtr()
                surface_tensor = pnvc.makefromDevicePtrUint8(surfPlane.GpuMem(), surfPlane.Width(), surfPlane.Height(), surfPlane.Pitch(), surfPlane.ElemSize())
                surface_tensor.resize_(3, target_h,target_w)
                
                try:
                    surface_tensor = torch.nn.functional.interpolate(surface_tensor.unsqueeze(0),resize).squeeze(0)
                except:
                    raise Exception("Surface tensor shape:{} --- resize shape: {}".format(surface_tensor.shape,resize))
            
                # This is optional and depends on what you NN expects to take as input
                # Normalize to range desired by NN. Originally it's 
                surface_tensor = surface_tensor.type(dtype=torch.cuda.FloatTensor)/255.0
                
                
                # apply normalization
                surface_tensor = F.normalize(surface_tensor,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                
                frame = (surface_tensor,pkt.pts)
                q.put(frame)
            
        ### Get next file if there is one 
        # sort directory files (by timestamp)
        files = os.listdir(directory)
        
        # filter out non-video_files and sort video files
        files = list(filter(  (lambda f: True if ".mkv" in f else False) ,   files))
        files.sort()
        
        # select next file that comes sequentially after last_file
        NEXTFILE = False
        for file in files:
            if file > last_file:
                last_file = file
                NEXTFILE = True           
                logger.debug("Loading frames from {}".format(file))
                break

        
        if not NEXTFILE:
            logger.warning("Loader ran out of input.")
            raise Exception("Reached last file for directory {}".format(directory))
            