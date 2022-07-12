import torch
from torchvision.transforms import functional as F
from torchvision.ops import roi_align, nms


def im_nms(detections,scores,threshold = 0.8,groups = None):
	"""
	Performs non-maximal supression on boxes given in image formulation
	detections - [d,8,2] array of boxes in state formulation
	scores - [d] array of box scores in range [0,1]
	threshold - float in range [0,1], boxes with IOU overlap > threshold are pruned
	groups - None or [d] tensor of unique group for each box, boxes from different groups will supress each other
	returns - idxs - list of indexes of boxes to keep
	"""

	minx = torch.min(detections[:,:,0],dim = 1)[0]
	miny = torch.min(detections[:,:,1],dim = 1)[0]
	maxx = torch.max(detections[:,:,0],dim = 1)[0]
	maxy = torch.max(detections[:,:,1],dim = 1)[0]

	boxes = torch.stack((minx,miny,maxx,maxy),dim = 1)

	if groups is not None:
	    large_offset = 10000
	    offset = groups.unsqueeze(1).repeat(1,4) * large_offset
	    boxes = boxes + large_offset

	idxs = nms(boxes,scores,threshold)
	return idxs

def space_nms(detections,scores,threshold = 0.1):
        """
        Performs non-maximal supression on boxes given in state formulation
        detections - [d,8,3] array of boxes in  space formulation
        scores - [d] array of box scores in range [0,1]
        threshold - float in range [0,1], boxes with IOU overlap > threshold are pruned
        returns - idxs - indexes of boxes to keep
        """
        
        # convert into xmin ymin xmax ymax form        
        boxes_new = torch.zeros([detections.shape[0],4],device = detections.device)
        boxes_new[:,0] = torch.min(detections[:,0:4,0],dim = 1)[0]
        boxes_new[:,2] = torch.max(detections[:,0:4,0],dim = 1)[0]
        boxes_new[:,1] = torch.min(detections[:,0:4,1],dim = 1)[0]
        boxes_new[:,3] = torch.max(detections[:,0:4,1],dim = 1)[0]
                
        idxs = nms(boxes_new,scores,threshold)
        return idxs
