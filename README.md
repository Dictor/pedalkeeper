## How to use 
## 1. 
```bash 
mkdir bag scene video 
``` 
- **bag:**  A directory to store *.bag files from the dataset. Bag files will be converted to scene files. 
- **scene:** A directory to store *.json scene files. Scene files contain decoded event arrays. 
- **video:** When `generate_train_set_video=True`, scenes will be converted into viewable MP4 video files. 

## 2. 
Download bag files from the dataset webpage and place them into the `bag` directory. 

## 3. 
Fill in pedal scene data using the following procedure. Pedal scene data is formatted as JSON and contains pedal pressing data with timestamps. 
``` 
a.  You downloaded a bag file to `./bag/back6.bag`. The dataset name is 'back6'. 
b.  The corresponding scene file for 'back6' will be located at `./scene/video_scene_back6.json` after executing `test.py`. 
c.  Manually fill in pedal data to `./scene/pedal_scene_back6.json`. 
d.  Execute `test.py` for training and verification. 
``` 

Example pedal JSON file content: 

```json 
[{"sec": 0.0, "pedal": 0}, {"sec": 4.0, "pedal": 1}, {"sec": 10.0, "pedal": 0}, {"sec": 22.0, "pedal": 0.5}, {"sec": 29.0, "pedal": 1}] 
``` 
This translates to: 

``` 
t    |   pedal 
-------------- 
0    | 0 
1    | 0 
2    | 0 
3    | 0 
4    | 1 
5    | 1 
...  | ... 
10   | 0 
11   | 0 
...  | ... 
``` 