import torch
import os
os.environ["user_config_directory"] = "/home/worklab/Documents/i24/i24_track/config"

from .src.track.TrackState import TrackState

print("\nMODULE: TrackState")
PASS = True

#%% Test TrackState intialization

try:
    TrackState()
    print("Test 1 PASS: Initializes correctly")
except:
    print("Test 1 FAIL: Error during initialization")
    PASS = False
    
    
    
    

#%% test TrackState add

# generate some dummy object data - need detections, directions, labels, times and scores
times = torch.rand(10)
detections = torch.rand(10,5)
directions  = torch.rand(10).round() *2 -1
labels = (torch.rand(10)*7).round()
scores = torch.rand(10)

try:
    tstate = TrackState()
    tstate.add(detections,directions,times,labels,scores)
    assert len(tstate.kf.obj_idxs) == 10
    print("Test 2 PASS: Adds objects")

except Exception as e:
    print("Test 2 FAIL: Error adding objects to TrackState: {}".format(e))
    PASS = False




#%% test TrackState __call__()

try:
    tstate()
    print("Test 3 PASS: calls correctly in default case")

except Exception as e:
    print("Test 3 FAIL: Error calling tstate() in default case: {}".format(e))
    PASS = False


try:
    tstate(mode = "dict")
    print("Test 4 PASS: calls correctly when return type = dict")

except Exception as e:
    print("Test 4 FAIL: Error calling tstate() when return type = dict: {}".format(e))
    PASS = False

try:
    #print(tstate(target_time = 2, with_direction = True))
    print("Test 5 PASS: calls correctly with optional arguments specified")

except Exception as e:
    print("Test 5 FAIL: Error calling tstate() with optional arguments specified: {}".format(e))
    PASS = False




#%% test TrackState predict()

try:
    tstate.predict(dt = torch.rand(10))
    #print(tstate())
    print("Test 6 PASS: predict() executes successfully")

except Exception as e:
    print("Test 6 FAIL: Error in predict(): {}".format(e))
    PASS = False




#%% test TrackState update()
try:
    ids = [1,3,5,7]
    detections = torch.rand(4,5)
    labels = (torch.rand(4)*8).round()
    scores = torch.rand(4)
    #print(tstate())
    tstate.update(detections,ids,labels,scores)
    print("Test 7 PASS: update() executes successfully")
    #print(tstate._history)
except Exception as e:
    print("Test 7 FAIL: Error in update(): {}".format(e))
    PASS = False





#%% test TrackState remove()

try:
    ids = [2,3]
    out = tstate.remove(ids)
    #print(out)
    assert len(out) == 2
    assert len(tstate.kf.obj_idxs) == 8
    print("Test 8 PASS: remove() executes successfully")

except Exception as e:
    print("Test 8 FAIL: Error in remove(): {}".format(e))
    PASS = False


    





#%% Conclude
if PASS:
    print("__________ All module tests passed. ___________")
else:
    print("__________ FAIL: One or more module tests failed __________")