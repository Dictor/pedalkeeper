from pathlib import Path
import orjson
import tqdm
import math
import sys

from rosbags.highlevel import AnyReader
import numpy as np


def GetEventsFromRosBag(expected_bag_path, expected_json_path):
  rb = []
  
  if Path(expected_json_path).exists():
    print("[GetEventsFromRosBag] {} has been converted before".format(expected_json_path))
    with open(expected_json_path, 'rb') as f:
      rb = orjson.loads(f.read())
      
  else:
    print("[GetEventsFromRosBag] {} haven't been converted yet, start converting".format(expected_bag_path))
    rb = decodeRosBag(expected_bag_path)
    with open(expected_json_path, 'wb') as f:
      f.write(orjson.dumps(rb))
      
  print("[GetEventsFromRosBag] {} events read from {}".format(len(rb), expected_json_path))
  return rb

def EventsToScene(events, size=(640, 480)):
  max_timestamp = -1
  min_timestamp = sys.maxsize
  for event in tqdm.tqdm(events):
    if event['ts'] > max_timestamp:
      max_timestamp = event['ts']
    if event['ts'] < min_timestamp:
      min_timestamp = event['ts']
  print("[EventsToScene] timestamp range is {} ~ {}".format(min_timestamp, max_timestamp))

  scene = []
  frame_speed = 1000000000.0 / 60.0
  frame_count_end = math.floor(max_timestamp / frame_speed) + 1
  frame_count_start = math.floor(min_timestamp / frame_speed)
  frame_count = frame_count_end - frame_count_start + 1
  print("[EventsToScene] total frame count will be {}".format(frame_count))
  for i in range(frame_count):
    scene.append(np.zeros((size[1], size[0]), dtype=int))

  for event in tqdm.tqdm(events):
    scene[math.floor(event['ts'] / frame_speed) - frame_count_start][size[1]-event['y']-1][size[0]-event['x']-1] = 1 if event['polarity'] else 0
      
  print("[EventsToScene] {} frames are made".format(len(scene)))
  return scene

# data key must follow time order
# data e.g.: [{'sec': 0.0, 'pedal': 1}, {'sec': 5.0, 'pedal': 0}]
def PedalToScene(data, scene_length, fps):
  scene = []
  for i in range(scene_length):
    scene.append(0.0)
  
  last_i = 0
  for i in range(scene_length):
    sec = i / fps
    if last_i < len(data)-1:
      if data[last_i+1]['sec'] <= sec:
        last_i += 1
    scene[i] = data[last_i]['pedal']
    
  return scene
      
    

def decodeRosBag(path, topic='/dvs/cam1/events'):
    bagpath = Path(path)
    result = []
    print("[decodeRosBag] start decoding {}".format(path))
    with AnyReader([bagpath]) as reader:
        connections = [x for x in reader.connections if x.topic == topic]
        for connection, timestamp, rawdata in tqdm.tqdm(reader.messages(connections=connections)):
            msg = reader.deserialize(rawdata, connection.msgtype)
            for event in msg.events:
              result.append({
                "x": event.x,
                "y": event.y,
                "polarity": event.polarity,
                "ts": event.ts.sec * 1000000000 + event.ts.nanosec
              })
    
    print("[decodeRosBag] complete decoding {}".format(path))
    return result
          
