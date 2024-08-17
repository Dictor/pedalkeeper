from rosbag import GetEventsFromRosBag, EventsToScene, PedalToScene
from video import ArrayToMp4
from train import Train, mobilevit_pedalkeeper, Verify
import os
import json
import numpy as np
import torch

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

# train set setting
train_dataset_name = ["back6"]
train_pedal_scene = [
  [{'sec': 0.0, 'pedal': 0}, {'sec': 7.0, 'pedal': 1}],
  
]
generate_train_set_video = False

# verify set setting
verify_dataset_name = "street2"
verify_video_scene = getVideoScene(verify_dataset_name)
verify_pedal_scene = PedalToScene([{'sec': 0.0, 'pedal': 0}, {'sec': 10.0, 'pedal': 1}], len(verify_video_scene), 60)


if generate_train_set_video:
  for name in train_dataset_name:
    ArrayToMp4(name, train_pedal_scene, "{name}.mp4", 60)

if os.path.exists("./model.save"):
  print("checkpoint found")
  model = mobilevit_pedalkeeper()
  model.load_state_dict(torch.load("./model.save"))
else:
  print("train and generate checkpoint")
  model = mobilevit_pedalkeeper()
  
  for i in range(len(train_dataset_name)):
    train_video_scene = getVideoScene(train_dataset_name[i])
    model = Train(model, train_video_scene, PedalToScene(train_pedal_scene[i], len(train_video_scene), 60), 20)
  
  torch.save(model.state_dict(), "./model.save")

Verify(model, verify_video_scene, verify_pedal_scene)