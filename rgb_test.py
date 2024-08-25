from rosbag import GetEventsFromRosBag, EventsToScene, PedalToScene
from video import ArrayToMp4
from train import Train, mobilevit_pedalkeeper, Verify
import os
import orjson
import numpy as np
import torch

import cv2

def read_mp4_to_numpy(filename):
    """
    mp4 비디오 파일을 numpy array로 읽는 함수

    Args:
        filename (str): mp4 비디오 파일 경로

    Returns:
        frames (numpy.ndarray): 비디오 프레임들을 담은 numpy array (shape: (프레임 수, 높이, 너비, 채널 수))
        fps (float): 비디오의 초당 프레임 수 (frames per second)
    """

    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return np.transpose(np.array(frames), (0, 3, 1, 2)) , fps

def getVideoScene(scene_name):
  video_scene = {}
  video_scene_path = "./rgb_video/{}.mp4".format(scene_name)
  video_scene, fps = read_mp4_to_numpy(video_scene_path)
  
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