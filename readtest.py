from rosbag import GetEventsFromRosBag, EventsToScene, PedalToScene
from video import ArrayToMp4
from train import Train, mobilevit_pedalkeeper, Verify
import os
import json
import numpy as np
import torch

train_dataset_name = "back6"
verify_dataset_name = "street2"

def getVideoScene(scene_name):
  video_scene = {}
  video_scene_path = "./scene/video_scene_{}.json".format(scene_name)
  if os.path.exists(video_scene_path):
    print("found video scene file {}".format(video_scene_path))
    with open(video_scene_path, 'r') as f:
      video_scene = json.load(f)
    video_scene = np.array(video_scene)
  else:
    print("generate video scene file {}".format(video_scene_path))
    events = GetEventsFromRosBag('./bag/{}.bag'.format(scene_name), './bag/{}.json'.format(scene_name))
    video_scene = EventsToScene(events)
    
    converted_video_scene = []
    for frame in video_scene:
      converted_video_scene.append(frame.tolist())
      
    with open(video_scene_path, 'w') as f:
      json.dump(converted_video_scene, f)
  
  return video_scene

train_video_scene = getVideoScene(train_dataset_name)
train_pedal_scene = PedalToScene([{'sec': 0.0, 'pedal': 0}, {'sec': 7.0, 'pedal': 1}], len(train_video_scene), 60) 
verify_video_scene = getVideoScene(verify_dataset_name)
verify_pedal_scene = PedalToScene([{'sec': 0.0, 'pedal': 0}, {'sec': 10.0, 'pedal': 1}], len(verify_video_scene), 60)

#ArrayToMp4(verify_video_scene, verify_pedal_scene, "output.mp4", 60)
if os.path.exists("./model.save"):
  print("checkpoint found")
  model = mobilevit_pedalkeeper()
  model.load_state_dict(torch.load("./model.save"))
else:
  print("train and generate checkpoint")
  model = Train(train_video_scene, train_pedal_scene, 3)
  torch.save(model.state_dict(), "./model.save")

Verify(model, verify_video_scene, verify_pedal_scene)