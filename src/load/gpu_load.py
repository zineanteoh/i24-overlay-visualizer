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

        cam_sequences = {}        
        for file in os.listdir(directory):
            sequence = os.path.join(directory,file)
            cam_name = re.search("p\dc\d",sequence).group(0)
            cam_sequences[cam_name] = sequence
        
        self.dummy_ts = 0
        
        # device loader is a list of lists, with list i containing all loaders for device i (hopefully in order but not well enforced by dictionary so IDK)
        self.device_loaders = [[] for i in range(torch.cuda.device_count())]
        for key in cam_sequences.keys():
            dev_id = self.cam_devices[key]
            sequence = cam_sequences[key]
            loader = GPUBackendFrameGetter(sequence,dev_id,ctx,resize = resize)
            
            self.device_loaders[dev_id].append(loader)
        
        
            
    def get_frames(self,target_time):
        
        frames = [[] for i in range(torch.cuda.device_count())]
        timestamps = []
        for dev_idx,this_dev_loaders in enumerate(self.device_loaders):
            for loader in this_dev_loaders:
                frame,ts = next(loader)
                frames[dev_idx].append(frame)
                timestamps.append(ts)
            
        # stack each list
        out = []
        for lis in frames:
            out.append(torch.stack(lis))
        timestamps = torch.tensor(timestamps)
        
        timestamps = timestamps * 0 + self.dummy_ts
        self.dummy_ts += 1/30.0
        
        return out,timestamps
    
class GPUBackendFrameGetter:
    def __init__(self,file,device,ctx,buffer_size = 5,resize = (1920,1080)):
        
        # create shared queue
        self.queue = ctx.Queue()
        self.frame_idx = -1
        self.device = device  
        
        self.file = file
        self.worker = ctx.Process(target=load_queue_vpf, args=(self.queue,file,device,buffer_size,resize))
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
        ts = frame[1] #/ 10e9
        im = frame[0]
        
        return im,ts
        
        if False: #TODO - implement shutdown
            self.worker.terminate()
            self.worker.join()
            return None
        
def load_queue_vpf(q,file,device,buffer_size,resize):
    resize = (resize[1],resize[0])
    gpuID = device
    device = torch.cuda.device("cuda:{}".format(gpuID))
    
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
    
    while True:
        if q.qsize() < buffer_size:
            pkt = nvc.PacketData()
            
            # Obtain NV12 decoded surface from decoder;
            rawSurface = nvDec.DecodeSingleSurface(pkt)
            if (rawSurface.Empty()):
                break
            
            # Convert to RGB interleaved;
            rgb_byte = to_rgb.Execute(rawSurface, cc_ctx)
        
            # Convert to RGB planar because that's what to_tensor + normalize are doing;
            rgb_planar = to_planar.Execute(rgb_byte, cc_ctx)
        
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
            
            # TODO - I think the model is trained with inverted RGB channels unfornately so this is a temporary fix
            #surface_tensor = surface_tensor[[2,1,0],:,:] 

            
            frame = (surface_tensor,pkt.pts)
            q.put(frame)