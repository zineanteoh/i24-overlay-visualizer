import cv2
import cv2 as cv
import time
import os
import numpy as np 
import csv

class Axis_Labeler():
    def __init__(self,directory, classes = ["Along Lane (green)", "Perpendicular to Lane (blue)" , "Up (red)"],ds = 1,direction = "EB"):
        self.frame = 1
        self.ds = ds
        
        self.directory = directory
        
        videos = os.listdir(directory)
        videos.sort()
        
        seq = os.path.join(directory,videos[0])
        
        self.cap = cv2.VideoCapture(seq)
        ret,frame = self.cap.read()
        if self.ds != 1:
            frame = cv2.resize(frame,(frame.shape[1]//self.ds,frame.shape[0]//self.ds))
        self.frames = [frame]
        
        
        self.axes = []
        
        #self.load_annotations()
        
        self.cur_image = self.frames[0]
        
        self.start_point = None # used to store click temporarily
        self.clicked = False 
        self.new = None # used to store a new box to be plotted temporarily
        self.cont = True
        self.define_direction = False
        
        # classes
        self.cur_class = 2
        self.n_classes = len(classes)
        self.class_names = classes
        self.colors = (np.random.rand(self.n_classes,3))*255
        self.colors[0] = np.array([0,255,0])
        self.colors[1] = np.array([255,0,0])
        self.colors[2] = np.array([0,0,255])

        self.plot_axes()
        self.changed = False
        self.direction = direction

    # def load_annotations(self):
    #     try:
    #         self.cur_frame_boxes = []
    #         name = "annotations/new/{}.csv".format(self.frames[self.frame-1].split("/")[-1].split(".")[0])
    #         with open(name,"r") as f:
    #             read = csv.reader(f)
    #             for row in read:
    #                 if len(row) == 5:
    #                     row = [int(float(item)) for item in row]
    #                 elif len(row) > 5:
    #                     row = [int(float(item)) for item in row[:5]] + [float(item) for item in row[5:]]
    #                 self.cur_frame_boxes.append(np.array(row))
                    
    #     except FileNotFoundError:
    #         self.cur_frame_boxes = []
        
    def plot_axes(self):
        self.cur_image = self.frames[self.frame-1].copy()

        last_source_idx = 0
        for box in self.axes:
                self.cur_image = cv2.line(self.cur_image,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),self.colors[int(box[4])],2)        
                
    def toggle_class(self):
        self.cur_class = (self.cur_class + 1) % self.n_classes
        print("Active Class: {}".format(self.class_names[self.cur_class]))
        
    def on_mouse(self,event, x, y, flags, params):
    
       if event == cv.EVENT_LBUTTONDOWN and not self.clicked:
         self.start_point = (x,y)
         self.clicked = True
         self.changed = True
         
       elif event == cv.EVENT_LBUTTONUP:
            box = np.array([self.start_point[0],self.start_point[1],x,y,self.cur_class]).astype(int)
            self.axes.append(box)
            self.new = box
            self.clicked = False
              
                 
    def next(self):
        self.clicked = False
        self.define_direction = False
        
        if self.frame == len(self.frames):
            ret,frame = self.cap.read()
            if ret:
                if self.ds != 1:
                    frame = cv2.resize(frame,(frame.shape[1]//self.ds,frame.shape[0]//self.ds))
                self.frames.append(frame)
            else:
                print("Last Frame.")    
                return
            
        self.frame += 1
        
        # load image and plot existing boxes
        self.cur_image = self.frames[self.frame-1].copy()
        self.plot_axes()
        self.changed = False
                
                
            
    def quit(self):
        
        cv2.destroyAllWindows()
        self.cont = False
        self.save()
        
    def save(self):
        if len(self.axes) == 0:
            return
        sequence = self.directory.split("/")[-1]
        name = sequence + "_{}_axes.csv".format(self.direction)
        if os.path.exists(name): # don't overwrite
            overwrite = input("Overwrite existing file? (y/n)")
            if overwrite != "y":
                return
        
        outputs = []
        for item in self.axes:
            output = list(item)
            outputs.append(output)
        
        with open(name,"w") as f:
            writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            writer.writerows(outputs)
        print("Saved axes as file {}".format(name))
        
        name = name.split(".csv")[0] +  ".png"
        cv2.imwrite(name,self.cur_image)
        
    def save_as(self,path):
        outputs = []
        for item in self.axes:
            output = list(item)
            outputs.append(output)
        
        with open(path,"w") as f:
            writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            writer.writerows(outputs)
        print("Saved axes as file {}".format(path))
            
    def clear(self):
        self.axes = []
        self.cur_image = self.frames[self.frame-1].copy()
        self.plot_axes()
        
    def undo(self):
        self.clicked = False
        self.define_direction = False
        self.define_magnitude = False
        
        self.axes = self.axes[:-1]
        self.cur_image = self.frames[self.frame-1].copy()
        self.plot_axes()
        
        
    def run(self):  
        cv2.namedWindow("window")
        cv.setMouseCallback("window", self.on_mouse, 0)
           
        while(self.cont): # one frame
        
           if self.new is not None:
               self.plot_axes()
                    
           self.new = None
               
           cv2.imshow("window", self.cur_image)
           title = "{} toggle class (1), switch frame (8-9), clear all (c), undo(u),   quit (q), switch frame (8-9)".format(self.class_names[self.cur_class])
           cv2.setWindowTitle("window",str(title))
           
           key = cv2.waitKey(1)
           if key == ord('9'):
                self.next()
           # elif key == ord('8'):
           #      self.prev()
           elif key == ord('c'):
                self.clear()
           elif key == ord("q"):
                self.quit()
           elif key == ord("1"):
                self.toggle_class()
           elif key == ord("u"):
               self.undo()
           elif key == ord("d"):
               self.remove()
               
directory = "/home/derek/Data/cv/video/06-02-2022/batch4"
for folder in os.listdir(directory):
    folder = os.path.join(directory,folder)
    
    axl = Axis_Labeler(folder,direction = "WB")
    axl.run()