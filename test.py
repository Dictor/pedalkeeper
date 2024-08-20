from rosbag import GetEventsFromRosBag, EventsToScene, PedalToScene
from video import ArrayToMp4
from train import Train, mobilevit_pedalkeeper, Verify
import os
import orjson
import numpy as np
import torch

def getVideoScene(scene_name):
  video_scene = {}
  video_scene_path = "./scene/video_scene_{}.json".format(scene_name)
  if os.path.exists(video_scene_path):
    print("[getVideoScene] found video scene file {}".format(video_scene_path))
    with open(video_scene_path, 'rb') as f:
      video_scene = orjson.loads(f.read())
    video_scene = np.array(video_scene)
  else:
    print("[getVideoScene] generate video scene file {}".format(video_scene_path))
    events = GetEventsFromRosBag('./bag/{}.bag'.format(scene_name), './bag/{}.json'.format(scene_name))
    video_scene = EventsToScene(events)
    
    converted_video_scene = []
    for frame in video_scene:
      converted_video_scene.append(frame.tolist())
      
    with open(video_scene_path, 'wb') as f:
      f.write(orjson.dumps(converted_video_scene))
  
  print("[getVideoScene] scene file {} finished".format(video_scene_path))
  return video_scene

def getPedalData(dataset_name):
  pedal_scene_path = "./scene/pedal_scene_{}.json".format(dataset_name)
  pedal_scene = {}
  if os.path.exists(pedal_scene_path):
    print("[getPedalData] found pedal scene file {}".format(dataset_name))
    with open(pedal_scene_path, 'rb') as f:
      pedal_scene = orjson.loads(f.read())
  else:
    print("[getPedalData] cannot found pedal scene file {}".format(dataset_name))
  
  return pedal_scene

# train set setting
train_dataset_name = ["back6"]#, 'back8','back10','sun5','sun14','sun15']
generate_train_set_video = True

# verify set setting
verify_dataset_name = "street2"
verify_video_scene = getVideoScene(verify_dataset_name)
verify_pedal_scene = PedalToScene(getPedalData(verify_dataset_name), len(verify_video_scene), 60)


if generate_train_set_video:
  for name in train_dataset_name:
    path = f"./video/{name}.mp4"
    if not os.path.exists(path):
      video = getVideoScene(name)
      pedal = getPedalData(name)
      ArrayToMp4(video, PedalToScene(pedal, len(video), 60), path, 60)

if os.path.exists("./model.save"):
  print("[test] checkpoint found")
  model = mobilevit_pedalkeeper()
  model.load_state_dict(torch.load("./model.save"))
else:
  print("[test] train and generate checkpoint")
  model = mobilevit_pedalkeeper()
  
  for i in range(len(train_dataset_name)):
    train_video_scene = getVideoScene(train_dataset_name[i])
    train_pedal_scene = getPedalData(train_dataset_name[i])
    model = Train(model, train_video_scene, PedalToScene(train_pedal_scene, len(train_video_scene), 60), 2)
  
  torch.save(model.state_dict(), "./model.save")

Verify(model, verify_video_scene, verify_pedal_scene)